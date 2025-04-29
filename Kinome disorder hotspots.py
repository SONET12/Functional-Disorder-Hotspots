import pandas as pd
import re
from collections import defaultdict

# Load the Excel sheets
doc1 = pd.read_excel(r"C:\Users\sonus\OneDrive\Desktop\Disorder Manuscript\mansucript datasets\kinase_data_fordisorder\kinase_data_fordisorder\Kinase_data.xlsx")
doc2 = pd.read_excel(r"C:\Users\sonus\OneDrive\Desktop\Disorder Manuscript\mansucript datasets\kinase_data_fordisorder\kinase_data_fordisorder\kinase_data_phosphorylation&mutation.xlsx")

# Define functional motifs
functional_motifs = ['DFG', 'HRD', 'DLG', 'GLD', 'APE', 'DRH', 'GFD']

# Function to parse position strings
def parse_position(pos_str):
    """
    Parse a position string into start and end integers.
    Returns (None, None) if parsing fails.
    """
    if pd.isna(pos_str) or not isinstance(pos_str, str) or not pos_str.strip():
        return None, None
    
    numbers = re.findall(r'\d+', pos_str)
    if len(numbers) == 2:
        return int(numbers[0]), int(numbers[1])
    elif len(numbers) == 1:
        return int(numbers[0]), int(numbers[0])
    
    return None, None  # Ensure a valid return format

# Function to check proximity
def is_near_region(feature_start, feature_end, region_start, region_end, buffer=10):
    """
    Check if a feature (motif, domain, etc.) is within the buffer zone of a region.
    Returns False if any input is None.
    """
    if None in (feature_start, feature_end, region_start, region_end):
        return False  # Skip invalid comparisons

    return (feature_end >= region_start - buffer) and (feature_start <= region_end + buffer)

# Initialize data structures
results = []
feature_counts = defaultdict(int)
max_features = 0

# Process each kinase and disordered region
for _, row1 in doc1.iterrows():
    kinase = row1['Kinase Name']
    dis_region = row1['Disordered Region']
    region_pos = row1['Region Position']
    region_start, region_end = parse_position(region_pos)
    
    if region_start is None:
        continue  # Skip if the region position is missing
    
    # Initialize feature count
    feature_count = 0
    
    # Check for functional motifs
    for motif in functional_motifs:
        if motif in row1 and pd.notna(row1[motif]):
            motif_start, motif_end = parse_position(row1[motif])
            if motif_start is not None and motif_end is not None:
                if is_near_region(motif_start, motif_end, region_start, region_end):
                    feature_count += 1
    
    # Check for kinase domain, binding sites, active sites
    for feature in ['Kinase Domain', 'Binding site', 'Active Site']:
        feature_start, feature_end = parse_position(row1.get(feature, ''))
        if feature_start is not None and feature_end is not None:
            if is_near_region(feature_start, feature_end, region_start, region_end):
                feature_count += 1
    
    # Match phosphorylation and mutation data
    doc2_matches = doc2[(doc2['Kinase Name'] == kinase) & (doc2['Disordered Region'] == dis_region)]
    if not doc2_matches.empty:
        row2 = doc2_matches.iloc[0]
        
        # Phosphorylation sites
        phos_sites = row2.get('Phosphorylation', '')
        if isinstance(phos_sites, str):  # Ensure it's a string before splitting
            for site in phos_sites.split(';'):
                phos_start, phos_end = parse_position(site)
                if phos_start is not None and phos_end is not None:
                    if is_near_region(phos_start, phos_end, region_start, region_end):
                        feature_count += 1
        
        # Mutation sites
        mutations = row2.get('Mutation', '')
        if isinstance(mutations, str):  # Ensure it's a string before splitting
            for mut in mutations.split(';'):
                mut_start, mut_end = parse_position(mut)
                if mut_start is not None and mut_end is not None:
                    if is_near_region(mut_start, mut_end, region_start, region_end):
                        feature_count += 1
    
    # Store feature count
    feature_counts[(kinase, dis_region, region_pos)] = feature_count
    max_features = max(max_features, feature_count)

# Normalize scores and identify hotspots
for (kinase, dis_region, region_pos), count in feature_counts.items():
    score = count / max_features if max_features > 0 else 0
    hotspot = 'Yes' if score > 0 else 'No'
    results.append({
        'Kinase Name': kinase,
        'Disordered Region': dis_region,
        'Position': region_pos,
        'Score': round(score, 2),
        'Hotspot': hotspot
    })

# Create DataFrame and save to Excel
output_df = pd.DataFrame(results)
output_df.to_excel('functional_disorder_hotspot_scores.xlsx', index=False)

print("Output saved to 'functional_disorder_hotspot_scores.xlsx'")
