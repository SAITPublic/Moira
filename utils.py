import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import os

def export_base_distribution(model, filename, min_pow=-24, max_pow=1, max_wires=10):
    powers = list(range(max_pow, min_pow - 1, -1))
    results = []

    linear_layers = {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear) and "lm_head" not in name}
    for name, layer in tqdm(linear_layers.items(), desc="Extracting base frequencies"):
        w = layer.weight.data
        layer_data = {
            "Layer": name,
            "Total_Params": w.numel(),
            "Pruned_(0_Wires)": (w == 0).sum().item(),
            "Total_Wires_Used": 0
        }
        for p in powers:
            layer_data[f"2^{p}"] = 0
        
        residual = w.clone()
        for _ in range(max_wires):
            abs_res = torch.abs(residual)
            active_mask = abs_res > 1e-12
            if not active_mask.any():
                break

            p = torch.round(torch.log2(torch.clamp(abs_res, min=1e-12)))
            p = torch.clamp(p, min=min_pow, max=max_pow)

            term = torch.sign(residual) * (2.0 ** p)
            term = torch.where(active_mask, term, torch.zeros_like(term))

            p_active = p[active_mask].long()
            shifted_p = p_active - min_pow
            bincounts = torch.bincount(shifted_p, minlength=(max_pow - min_pow + 1))

            for idx, count in enumerate(bincounts.tolist()):
                if count > 0:
                    actual_p = idx + min_pow
                    layer_data[f"2^{actual_p}"] += count
                    layer_data["Total_Wires_Used"] += count
            
            residual = residual - term
        
        layer_data["Avg_Wires_Per_Weight"] = layer_data["Total_Wires_Used"] / layer_data["Total_Params"]
        results.append(layer_data)
    
    df = pd.DataFrame(results)
    df.set_index("Layer", inplace=True)
    df_t = df.T.astype(float)

    powers_ascending = list(range(min_pow, max_pow + 1))
    basis_rows = [f"2^{p}" for p in powers_ascending]
    meta_rows = [ "Pruned_(0_Wires)", "Total_Wires_Used", "Total_Params", "Avg_Wires_Per_Weight"]
    df_t = df_t.reindex(basis_rows + meta_rows)
    df_t["Global_Total"] = df_t.sum(axis=1)

    total_params = float(df_t.loc["Total_Params", "Global_Total"])
    total_wires = float(df_t.loc["Total_Wires_Used", "Global_Total"])
    if total_params > 0:
        df_t.loc["Avg_Wires_Per_Weight"] = total_wires / total_params
    
    df_t.index.name = "Metric / Basis"
    df_t.reset_index(inplace=True)
    df_t.to_csv(filename, index=False)
    print(f"\nBase frequency data saved to '{filename}'.")
    return df_t

def export_results(file_path, results_dict):
    new_df = pd.DataFrame([results_dict])
    file_exists = os.path.isfile(file_path)
    new_df.to_csv(file_path, mode='a', index=False, header=not file_exists)    
    print(f"\n[Complete report] Data is summarized in {file_path}")