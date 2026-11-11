import torch
import torch.nn as nn
import gc

class Moira:
    def __init__(self, target_wires=1.0):
        self.target_wires = target_wires
        self.worker_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _forward_dummy(self, model, tokenizer, num_samples):
        text = "Artificial intelligence and machine learning are transforming the future of technology and society. "
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            for _ in range(num_samples):
                model(**inputs)

    def _forward_c4(self, model, c4_loader):
        input_ids = c4_loader.input_ids
        nsamples = input_ids.numel() // model.seqlen
        print(f"c4 nsamples {nsamples}")
        with torch.no_grad():
            for i in range(nsamples):
                if i % 50 == 0:
                    print(f"sample {i}")
                inputs = input_ids[:, i * model.seqlen:(i + 1) * model.seqlen].to(model.device)
                model(inputs)

    def _base_wire_stats(self, w, m_bits):
        m, e = torch.frexp(w)
        m = torch.abs(m)
        all_shifts = []
        for bit_idx in range(1, m_bits + 2):
            threshold = 2 ** (-bit_idx)
            mask = m >= threshold
            if mask.sum() == 0:
                continue
            active_shifts = e[mask] - bit_idx
            all_shifts.append(active_shifts)
            m[mask] -= threshold

        final_shifts = torch.cat(all_shifts)
        _, counts = torch.unique(final_shifts, return_counts=True)
        m_bit_count = counts.sum()
        max_wires = torch.ceil(m_bit_count / w.numel())
        return m_bit_count, max_wires

    def _fit_lambda(self, name, w, act, max_wires, iters, tol):
        target_avg_wires = self.target_wires
        total_elements_layer = w.numel()
        low_lambda = 0.0
        high_lambda = (torch.abs(w) * act).max().item()
        best_w_hat, best_k_tensor = None, None
        log_steps = []

        for step in range(iters):
            mid_lambda = (low_lambda + high_lambda) / 2.0
            current_w_hat, current_k = self.greedy_weight_factorization(max_wires, w, act, lambda_=mid_lambda)
            current_avg_wires = current_k.sum().item() / total_elements_layer

            print(f"Step {step+1:2d} | Lambda: {mid_lambda:.8f} | Avg Wires: {current_avg_wires:.4f}")
            log_steps.append({
                "layer": name,
                "step": step + 1,
                "mid_lambda": float(mid_lambda),
                "avg_wires": float(current_avg_wires),
            })

            if abs(current_avg_wires - target_avg_wires) < tol:
                return current_w_hat, current_k, log_steps

            if current_avg_wires > target_avg_wires:
                low_lambda = mid_lambda
            else:
                high_lambda = mid_lambda

            best_w_hat, best_k_tensor = current_w_hat, current_k
        return best_w_hat, best_k_tensor, log_steps

    def collect_activation_stats(self, model, tokenizer, c4_loader, num_samples=10, if_dummy=True):
        sums, counts = {}, {}
        def make_hook(name):
            def hook(module, inputs, output):
                x = inputs[0].detach().float()
                if x.dim() == 2:
                    act = x.abs()
                elif x.dim() == 3:
                    act = torch.norm(x.abs(), dim=1)
                if name in sums:
                    sums[name] += act
                    counts[name] += x.size(0)
                else:
                    sums[name] = act
                    counts[name] = x.size(0)
            return hook

        hooks = [module.register_forward_hook(make_hook(name))
                 for name, module in model.named_modules() if isinstance(module, nn.Linear)]

        print("Collecting Activation Statistics...")
        self._forward_dummy(model, tokenizer, num_samples) if if_dummy else self._forward_c4(model, c4_loader)
        for hook in hooks:
            hook.remove()
        print("Activation Collection Complete.")
        return {name: total / counts[name] for name, total in sums.items()}

    def greedy_weight_factorization(self, max_wires, w, act, lambda_, min_pow=-24, max_pow=1):
        if act.dim() == 1:
            act = act.view(1, -1).expand_as(w)

        best_w_hat = torch.zeros_like(w)
        best_cost = torch.abs(w) * act
        best_k = torch.zeros_like(w, dtype=torch.long)

        residual = w.clone()

        current_w_hat = torch.zeros_like(w)
        k = 1
        while k <= max_wires:
            abs_res = torch.abs(residual)
            abs_res_clamped = torch.clamp(abs_res, min=1e-12)
            p = torch.round(torch.log2(abs_res_clamped))
            p = torch.clamp(p, min=min_pow, max=max_pow)

            term = torch.sign(residual) * (2.0 ** p)
            term = torch.where(abs_res < 1e-12, torch.zeros_like(term), term)

            current_w_hat = current_w_hat + term
            residual = residual - term

            current_error = torch.abs(w - current_w_hat)
            current_cost = current_error * act + lambda_ * k

            better_mask = current_cost < best_cost
            if better_mask.sum() == 0:
                break

            best_cost = torch.where(better_mask, current_cost, best_cost)
            best_w_hat = torch.where(better_mask, current_w_hat, best_w_hat)
            best_k = torch.where(better_mask, torch.full_like(best_k, k), best_k)

            k += 1
        return best_w_hat, best_k

    def wire_budget_fitting(self, model, act_stats, m_bits=10, iters=30, tol=0.01):
        total_elements, total_wires_used = 0, 0
        log_steps = []

        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear) and "lm_head" not in name:
                print(f"=== Wire-budget fitting (Target: {self.target_wires} Wires/Weight)")
                print(f"[{name}] Applying Lossy Wiring...")
                if name in act_stats:
                    original_device = layer.weight.device
                    w = layer.weight.data.to(self.worker_device)
                    act = act_stats[name].to(self.worker_device)

                    _, max_wires = self._base_wire_stats(w, m_bits)
                    print(f"[{name}] Expected the maximum bounded wires: {max_wires}")

                    best_w_hat, best_k, layer_steps = self._fit_lambda(name, w, act, max_wires, iters, tol)
                    log_steps.extend(layer_steps)

                    layer.weight.data = best_w_hat.to(original_device)
                    total_elements += w.numel()
                    total_wires_used += best_k.sum().item()

                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"[{name}] Complete!")

        final_global_avg = total_wires_used / total_elements if total_elements > 0 else 0
        print("\n=== Application Summary ===")
        print(f"Total Wires Used: {total_wires_used:}")
        print(f"Final Global Avg Wires per Weight: {final_global_avg:.4f}")
        return total_wires_used, final_global_avg, log_steps

    def run(self, model, tokenizer, c4_loader):
        act_stats = self.collect_activation_stats(model, tokenizer, c4_loader, if_dummy=False)
        return self.wire_budget_fitting(model, act_stats)
