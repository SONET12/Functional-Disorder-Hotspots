#!/usr/bin/env python3
"""
Tkinter Interfaces for Predicting Mutation and Phosphorylation Impact on Disorder-to-Order Transition

Two interfaces:
1. Mutation Predictor: Analyzes the effect of a single amino acid mutation.
2. Phosphorylation Predictor: Evaluates the impact of phosphorylation at a specified site.
"""

import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import math
from collections import Counter

# =============================================================================
# Feature Extraction Functions
# =============================================================================

# Chou–Fasman parameters
alpha_propensity = {
    'A': 1.45, 'C': 0.77, 'D': 0.98, 'E': 1.53, 'F': 1.12,
    'G': 0.53, 'H': 1.24, 'I': 1.00, 'K': 1.07, 'L': 1.34,
    'M': 1.20, 'N': 0.73, 'P': 0.59, 'Q': 1.17, 'R': 0.79,
    'S': 0.79, 'T': 0.82, 'V': 1.14, 'W': 1.14, 'Y': 0.61
}
beta_propensity = {
    'A': 0.97, 'C': 1.30, 'D': 0.80, 'E': 0.26, 'F': 1.28,
    'G': 0.81, 'H': 0.71, 'I': 1.60, 'K': 0.74, 'L': 1.22,
    'M': 1.67, 'N': 0.65, 'P': 0.62, 'Q': 1.23, 'R': 0.90,
    'S': 0.72, 'T': 1.20, 'V': 1.65, 'W': 1.19, 'Y': 1.29
}

def sliding_window_feature(seq, scale_dict, window=7):
    n = len(seq)
    values = np.array([scale_dict.get(aa, 0.0) for aa in seq], dtype=float)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode='edge')
    avg_vals = np.zeros(n, dtype=float)
    for i in range(n):
        window_data = padded[i:i+window]
        if window_data.size > 0:
            avg_vals[i] = np.mean(window_data)
    return avg_vals

def secondary_structure_scores(seq, window=7):
    alpha_scores = sliding_window_feature(seq, alpha_propensity, window)
    beta_scores = sliding_window_feature(seq, beta_propensity, window)
    return {'alpha': alpha_scores, 'beta': beta_scores}

def aa_composition(seq):
    count = Counter(seq)
    total = len(seq)
    composition = {aa: count.get(aa, 0) / total for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    return composition

kyte_doolittle = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5,  'K': -3.9, 'L': 3.8,
    'M': 1.9,  'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3
}

def hydrophobicity_profile(seq, window=7):
    return sliding_window_feature(seq, kyte_doolittle, window)

def net_charge(seq, pH=7.0):
    positives = seq.count('K') + seq.count('R')
    negatives = seq.count('D') + seq.count('E')
    return positives - negatives

def charge_distribution(seq, window=7):
    n = len(seq)
    charges = np.zeros(n, dtype=float)
    for i in range(n):
        start = max(0, i - window//2)
        end = min(n, i + window//2 + 1)
        sub_seq = seq[start:end]
        charges[i] = net_charge(sub_seq)
    return charges

side_chain_volumes = {
    'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
    'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
    'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
    'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
}
flexibility_index = {
    'A': 0.360, 'C': 0.350, 'D': 0.510, 'E': 0.500, 'F': 0.310,
    'G': 0.540, 'H': 0.320, 'I': 0.460, 'K': 0.470, 'L': 0.370,
    'M': 0.300, 'N': 0.460, 'P': 0.510, 'Q': 0.490, 'R': 0.530,
    'S': 0.510, 'T': 0.440, 'V': 0.500, 'W': 0.310, 'Y': 0.420
}

def average_side_chain_volume(seq):
    volumes = [side_chain_volumes.get(aa, 0.0) for aa in seq]
    return np.mean(volumes) if volumes else 0.0

def average_flexibility(seq):
    flex_vals = [flexibility_index.get(aa, 0.0) for aa in seq]
    return np.mean(flex_vals) if flex_vals else 0.0

def shannon_entropy(seq, window=7):
    n = len(seq)
    entropy_profile = np.zeros(n, dtype=float)
    for i in range(n):
        start = max(0, i - window//2)
        end = min(n, i + window//2 + 1)
        sub_seq = seq[start:end]
        counts = Counter(sub_seq)
        total = len(sub_seq)
        if total > 0:
            entropy = -sum((count/total) * math.log2(count/total) for count in counts.values())
            entropy_profile[i] = entropy
    return entropy_profile

mj_potential = {
    ('A','A'): -0.5, ('A','C'): -0.2, ('A','D'): 0.3,
}
default_mj = -0.1

def get_mj_score(aa1, aa2):
    key = (aa1, aa2) if (aa1, aa2) in mj_potential else (aa2, aa1)
    return mj_potential.get(key, default_mj)

def contact_potential(seq, window=5):
    n = len(seq)
    potentials = np.zeros(n, dtype=float)
    for i in range(n):
        start = max(0, i - window//2)
        end = min(n, i + window//2 + 1)
        sub_seq = seq[start:end]
        score = 0.0
        count = 0
        for j in range(len(sub_seq)):
            for k in range(j+1, len(sub_seq)):
                score += get_mj_score(sub_seq[j], sub_seq[k])
                count += 1
        potentials[i] = score / count if count > 0 else 0.0
    return potentials

def free_energy_estimate(seq, hydro_weight=1.0, contact_weight=1.0, entropy_weight=0.5):
    hydro = np.mean([kyte_doolittle.get(aa, 0.0) for aa in seq]) if seq else 0.0
    contact = np.mean(contact_potential(seq, window=5)) if seq else 0.0
    entropy = np.mean(shannon_entropy(seq, window=7)) if seq else 0.0
    delta_G = hydro_weight * (-hydro) + contact_weight * contact + entropy_weight * entropy
    return delta_G

def estimate_folding_rate(seq, k0=1e6, alpha=0.01, temp_factor=0.6):
    L = len(seq)
    delta_G = free_energy_estimate(seq)
    k_f = k0 * np.exp(-alpha * L) * np.exp(-delta_G / temp_factor)
    return k_f

# =============================================================================
# Predictive Probability Model
# =============================================================================

def logistic_probability(delta_delta_G, scale=1.0):
    return 1.0 / (1.0 + math.exp(scale * delta_delta_G))

def predict_mutation_probability(wt_seq, mutation_pos, mutated_res, scale=1.0):
    if mutation_pos < 1 or mutation_pos > len(wt_seq):
        raise ValueError("Mutation position out of range.")
    if mutated_res not in 'ACDEFGHIKLMNPQRSTVWY':
        raise ValueError("Invalid mutated residue.")
    mutated_seq = wt_seq[:mutation_pos - 1] + mutated_res + wt_seq[mutation_pos:]
    delta_G_wt = free_energy_estimate(wt_seq)
    delta_G_mut = free_energy_estimate(mutated_seq)
    delta_delta_G = delta_G_mut - delta_G_wt
    probability = logistic_probability(delta_delta_G, scale)
    return probability, delta_G_wt, delta_G_mut, mutated_seq

# =============================================================================
# Phosphorylation Effect
# =============================================================================

phospho_sites = ['S', 'T', 'Y']
phospho_motifs = ['SP', 'TP', 'RXXS']

def phosphorylation_effect(seq, pos, kinase, window=7):
    if pos < 1 or pos > len(seq):
        raise ValueError("Phosphorylation position out of range.")
    if seq[pos-1] not in phospho_sites:
        raise ValueError("Position is not a valid phosphorylation site (S, T, or Y).")
    
    phospho_shift = 0.0
    start = max(0, pos - window//2 - 1)
    end = min(len(seq), pos + window//2)
    sub_seq = seq[start:end]
    
    for i in range(len(sub_seq)-1):
        dipep = sub_seq[i:i+2]
        if dipep in phospho_motifs and sub_seq[i] in phospho_sites:
            phospho_shift += 0.3
    if pos-4 >= 0 and seq[pos-4:pos-1] == 'RXX' and seq[pos-1] in phospho_sites:
        phospho_shift += 0.3
    
    phospho_shift += 0.5
    return phospho_shift

def predict_phosphorylation_probability(seq, pos, kinase, scale=1.0):
    delta_G_wt = free_energy_estimate(seq)
    phospho_shift = phosphorylation_effect(seq, pos, kinase)
    delta_G_phos = delta_G_wt - phospho_shift
    delta_delta_G = delta_G_phos - delta_G_wt
    probability = logistic_probability(delta_delta_G, scale)
    return probability, delta_G_wt, delta_G_phos

# =============================================================================
# Monte Carlo Simulation
# =============================================================================

def monte_carlo_simulation(seq, n_iter=1000):
    best_prob = 0.0
    for _ in range(n_iter):
        perturbed_seq = ''.join([
            seq[i] if np.random.rand() > 0.05 else np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'))
            for i in range(len(seq))
        ])
        delta_G = free_energy_estimate(perturbed_seq)
        prob = logistic_probability(delta_G)
        best_prob = max(best_prob, prob)
    return best_prob

# =============================================================================
# Tkinter GUI with Tabs
# =============================================================================

class TransitionPredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("Disorder-to-Order Transition Predictor")
        
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(pady=10, expand=True)
        
        self.mutation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.mutation_frame, text="Mutation Predictor")
        self.setup_mutation_tab()
        
        self.phospho_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.phospho_frame, text="Phosphorylation Predictor")
        self.setup_phospho_tab()

    def setup_mutation_tab(self):
        tk.Label(self.mutation_frame, text="Kinase Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.mut_entry_kinase = tk.Entry(self.mutation_frame, width=30)
        self.mut_entry_kinase.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(self.mutation_frame, text="Disorder Region Sequence:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.mut_entry_seq = tk.Entry(self.mutation_frame, width=30)
        self.mut_entry_seq.grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(self.mutation_frame, text="Mutation Position (1-indexed):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.mut_entry_pos = tk.Entry(self.mutation_frame, width=10)
        self.mut_entry_pos.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        tk.Label(self.mutation_frame, text="Mutated Residue:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.mut_entry_res = tk.Entry(self.mutation_frame, width=5)
        self.mut_entry_res.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        tk.Button(self.mutation_frame, text="Submit", command=self.submit_mutation).grid(row=4, column=0, columnspan=2, pady=10)
        
        self.mut_output_text = tk.Text(self.mutation_frame, height=10, width=50)
        self.mut_output_text.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

    def setup_phospho_tab(self):
        tk.Label(self.phospho_frame, text="Kinase Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.phos_entry_kinase = tk.Entry(self.phospho_frame, width=30)
        self.phos_entry_kinase.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(self.phospho_frame, text="Disorder Region Sequence:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.phos_entry_seq = tk.Entry(self.phospho_frame, width=30)
        self.phos_entry_seq.grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(self.phospho_frame, text="Phosphorylation Position (1-indexed):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.phos_entry_pos = tk.Entry(self.phospho_frame, width=10)
        self.phos_entry_pos.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        tk.Button(self.phospho_frame, text="Submit", command=self.submit_phospho).grid(row=3, column=0, columnspan=2, pady=10)
        
        self.phos_output_text = tk.Text(self.phospho_frame, height=10, width=50)
        self.phos_output_text.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

    def submit_mutation(self):
        kinase = self.mut_entry_kinase.get().strip()
        seq = self.mut_entry_seq.get().strip().upper()
        pos_str = self.mut_entry_pos.get().strip()
        mutated_res = self.mut_entry_res.get().strip().upper()
        
        if not seq:
            messagebox.showerror("Error", "Please enter the disorder region sequence.")
            return
        if not all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in seq):
            messagebox.showerror("Error", "Sequence contains invalid amino acid codes.")
            return
        if not pos_str.isdigit():
            messagebox.showerror("Error", "Mutation position must be an integer.")
            return
        if len(mutated_res) != 1 or mutated_res not in 'ACDEFGHIKLMNPQRSTVWY':
            messagebox.showerror("Error", "Please enter a valid single-letter amino acid code.")
            return
        
        mutation_pos = int(pos_str)
        
        try:
            probability, delta_G_wt, delta_G_mut, mutated_seq = predict_mutation_probability(seq, mutation_pos, mutated_res)
            mc_prob = monte_carlo_simulation(mutated_seq)
            combined_prob = (probability + mc_prob) / 2
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
            return
        
        self.mut_output_text.delete(1.0, tk.END)
        result_text = f"Kinase: {kinase}\n"
        result_text += f"Wild-Type Sequence: {seq}\n"
        result_text += f"Mutated Sequence: {mutated_seq}\n"
        result_text += f"Mutation Position: {mutation_pos}\n"
        result_text += f"Mutated Residue: {mutated_res}\n\n"
        result_text += f"ΔG (Wild-Type): {delta_G_wt:.3f}\n"
        result_text += f"ΔG (Mutant): {delta_G_mut:.3f}\n"
        result_text += f"ΔΔG: {delta_G_mut - delta_G_wt:.3f}\n"
        result_text += f"Predicted Transition Probability: {combined_prob*100:.2f}%\n"
        self.mut_output_text.insert(tk.END, result_text)

    def submit_phospho(self):
        kinase = self.phos_entry_kinase.get().strip()
        seq = self.phos_entry_seq.get().strip().upper()
        pos_str = self.phos_entry_pos.get().strip()
        
        if not seq:
            messagebox.showerror("Error", "Please enter the disorder region sequence.")
            return
        if not all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in seq):
            messagebox.showerror("Error", "Sequence contains invalid amino acid codes.")
            return
        if not pos_str.isdigit():
            messagebox.showerror("Error", "Phosphorylation position must be an integer.")
            return
        
        pos = int(pos_str)
        
        try:
            probability, delta_G_wt, delta_G_phos = predict_phosphorylation_probability(seq, pos, kinase)
            mc_prob = monte_carlo_simulation(seq)
            combined_prob = (probability + mc_prob) / 2
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
            return
        
        self.phos_output_text.delete(1.0, tk.END)
        result_text = f"Kinase: {kinase}\n"
        result_text += f"Sequence: {seq}\n"
        result_text += f"Phosphorylation Position: {pos}\n"
        result_text += f"Residue: {seq[pos-1]}\n\n"
        result_text += f"ΔG (Wild-Type): {delta_G_wt:.3f}\n"
        result_text += f"ΔG (Phosphorylated): {delta_G_phos:.3f}\n"
        result_text += f"ΔΔG: {delta_G_phos - delta_G_wt:.3f}\n"
        result_text += f"Predicted Transition Probability: {combined_prob*100:.2f}%\n"
        self.phos_output_text.insert(tk.END, result_text)

# =============================================================================
# Main Application Launch
# =============================================================================

if __name__ == '__main__':
    root = tk.Tk()
    app = TransitionPredictorApp(root)
    root.mainloop()