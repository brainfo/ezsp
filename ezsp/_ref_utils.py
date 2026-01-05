def prepare_single_cell_reference(
    sc_path: str,
    n_top_genes: int = 100,
    groupby: str = "major_celltype",
    min_cells_per_type: int = 10,
    cell_types: set[str] | None = None,
    cell_types_proportion: dict[str, float] | None = None,
) -> tuple[AnnData, list[str]]:
    """Load and prepare single-cell reference data for Tangram."""
    adata_sc = sc.read_h5ad(sc_path)
    logger.info(f"Loaded single-cell reference: {adata_sc.shape}")
    
    # Filter cell types by minimum cell count
    celltype_counts = adata_sc.obs[groupby].value_counts()
    valid_celltypes = set(celltype_counts[celltype_counts >= min_cells_per_type].index.tolist()) & cell_types
    removed = celltype_counts[celltype_counts < min_cells_per_type]
    
    if len(removed) > 0:
        logger.warning(f"Filtering {len(removed)} cell types with < {min_cells_per_type} cells: {dict(removed)}")
    
    adata_sc = adata_sc[adata_sc.obs[groupby].isin(valid_celltypes)].copy()
    logger.info(f"After filtering: {adata_sc.shape}, {len(valid_celltypes)} cell types")
    if cell_types_proportion is not None:
        # Calculate available counts per cell type
        ct_counts = {}
        for ct in valid_celltypes:
            ct_mask = adata_sc.obs[groupby] == ct
            ct_counts[ct] = ct_mask.sum()
        
        # Find the limiting cell type using ratio: count / proportion
        # The cell type with the lowest ratio is the true bottleneck
        # This ensures all cell types can be downsampled (no upsampling needed)
        limiting_ratios = {
            ct: ct_counts[ct] / prop 
            for ct, prop in cell_types_proportion.items() 
            if prop > 0 and ct_counts[ct] > 0
        }
        
        if not limiting_ratios:
            logger.warning("No valid cell types for proportional sampling")
        else:
            limiting_ct = min(limiting_ratios, key=limiting_ratios.get)
            total_output = int(limiting_ratios[limiting_ct])
            logger.info(f"Limiting cell type: {limiting_ct} (ratio={limiting_ratios[limiting_ct]:.1f})")
            logger.info(f"Total output size: {total_output}")
            
            # Calculate target count for each cell type
            sampled_indices = []
            for ct, proportion in cell_types_proportion.items():
                ct_mask = adata_sc.obs[groupby] == ct
                ct_indices = adata_sc.obs_names[ct_mask].tolist()
                n_target = int(total_output * proportion)
                
                if n_target > 0 and len(ct_indices) > 0:
                    # Always downsampling since limiting_ct ensures n_target <= len(ct_indices)
                    sampled = np.random.choice(ct_indices, size=n_target, replace=False)
                    sampled_indices.extend(sampled)
                    logger.info(f"Sampled {n_target}/{len(ct_indices)} cells for {ct} (target proportion={proportion})")
            
            adata_sc = adata_sc[sampled_indices].copy()
            
            # Log actual proportions achieved
            actual_props = adata_sc.obs[groupby].value_counts(normalize=True)
            logger.info(f"After proportional sampling: {adata_sc.shape}")
            logger.info(f"Achieved proportions: {dict(actual_props)}")
    
    # Rank genes
    adata_sc.X = adata_sc.layers["log_norm"].copy()
    sc.tl.rank_genes_groups(adata_sc, groupby=groupby, use_raw=False)
    
    markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[:n_top_genes, :]
    marker_genes = np.unique(markers_df.melt().value.values).tolist()
    logger.info(f"Identified {len(marker_genes)} marker genes")
    
    adata_sc.X = adata_sc.layers["raw"].copy()
    return adata_sc, marker_genes