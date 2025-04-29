import pandas as pd

# Read the input file
data = pd.read_excel(r"C:\Users\sonus\OneDrive\Desktop\Disorder Manuscript\mansucript datasets\kinase_data_fordisorder\kinase_data_fordisorder\Kinase_family_final.xlsx")

# Parse Region Position into start and end
data['start'] = data['Region Position'].str.split('-').str[0].astype(int)
data['end'] = data['Region Position'].str.split('-').str[1].astype(int)

# Group by family
family_groups = data.groupby('Family')

# Store results
results = []

for family, group in family_groups:
    # Collect regions: [kinase, start, end, sequence]
    regions = group[['Kinase Name', 'start', 'end', 'Disordered Region']].values.tolist()
    
    # Sort regions by start position
    regions.sort(key=lambda x: x[1])  # x[1] is start
    
    # Cluster regions based on start positions within 10 residues
    clusters = []
    current_cluster = [regions[0]]
    for region in regions[1:]:
        if region[1] - current_cluster[-1][1] <= 10:
            current_cluster.append(region)
        else:
            # Check if cluster has at least two different kinases
            kinases_in_cluster = set(r[0] for r in current_cluster)
            if len(kinases_in_cluster) >= 2:
                clusters.append(current_cluster)
            current_cluster = [region]
    # Check the last cluster
    kinases_in_cluster = set(r[0] for r in current_cluster)
    if len(kinases_in_cluster) >= 2:
        clusters.append(current_cluster)
    
    # If no valid clusters, all kinases score 0
    if not clusters:
        for _, row in group.iterrows():
            results.append([row['Kinase Name'], row['Disordered Region'], row['Region Position'], family, 0.0000])
        continue
    
    # Approximate protein length as max end position
    L_protein = max(r[2] for r in regions)  # r[2] is end
    
    # Step 2: Compute max_L_min across clusters
    max_L_min = 0
    for cluster in clusters:
        sequences = [r[3] for r in cluster]
        L_min = min(len(seq) for seq in sequences)
        max_L_min = max(max_L_min, L_min)
    
    # Step 3: Score each cluster
    cluster_scores = {}
    for cluster in clusters:
        sequences = [r[3] for r in cluster]
        
        # SSS with padding
        L_max = max(len(seq) for seq in sequences)
        matches = 0
        for i in range(L_max):
            residues = [seq[i] if i < len(seq) else '-' for seq in sequences]
            if all(res == residues[0] and res != '-' for res in residues):
                matches += 1
        SSS = matches / L_max if L_max > 0 else 0
        
        # PSS (simplified)
        PSS = 1  # Assuming overlap
        
        # DRC
        L_min = min(len(seq) for seq in sequences)
        DRC = L_min / max_L_min if max_L_min > 0 else 0
        
        # Cluster score
        score = 0.3 * PSS + 0.5 * SSS + 0.2 * DRC
        cluster_scores[tuple(map(tuple, cluster))] = score  # Convert list of lists to tuple of tuples
    
    # Step 4: Assign kinase scores
    kinase_scores = {}
    for cluster, score in cluster_scores.items():
        for kinase, _, _, disorder_region in cluster:
            kinase_scores[(kinase, disorder_region)] = max(kinase_scores.get((kinase, disorder_region), 0), score)
    
    # Store results
    for (kinase, disorder_region), score in kinase_scores.items():
        region_position = group.loc[
    (group['Kinase Name'] == kinase) & (group['Disordered Region'] == disorder_region),
    'Region Position'
].values[0]

        results.append([kinase, disorder_region, region_position, family, round(score, 4)])

# Convert results to DataFrame
output_df = pd.DataFrame(results, columns=['Kinase', 'Disorder Region', 'Region Position', 'Family', 'Score'])

# Save to Excel
output_df.to_excel("Kinase_Disorder_Scorescorrect.xlsx", index=False)

# Print sample output
print(output_df.head())
print("Output saved to 'Kinase_Disorder Scores.xlsx'")