"""Entry point for ezsp package.

Usage:
    python -m ezsp tangram config.yaml    # Run Tangram pipeline
    python -m ezsp plot config.yaml       # Run plotting
"""
import argparse
import pathlib
import sys

from ._pp import load_config, setup_logging


def tangram_main(args):
    """Run Tangram deconvolution pipeline."""
    from .tg_adata import run_batch, run_pipeline

    config = load_config(args.config)
    
    batch_keys = ["sdata_path", "sc_path", "output_dir", "library_id", "image_key", "log_dir"]
    is_batch = isinstance(config.get("sdata_path"), list)
    
    if is_batch:
        lengths = {k: len(config[k]) for k in batch_keys if isinstance(config.get(k), list)}
        if len(set(lengths.values())) > 1:
            print(f"ERROR: Batch parameters must have same length: {lengths}")
            sys.exit(1)
        
        n_samples = list(lengths.values())[0]
        batch_params = []
        for i in range(n_samples):
            params = {
                "sdata_path": config["sdata_path"][i],
                "sc_path": config["sc_path"][i],
                "output_dir": config["output_dir"][i],
                "log_dir": config["log_dir"][i],
                "library_id": config["library_id"][i],
                "image_key": config["image_key"][i],
                **{k: config.get(k, v) for k, v in [
                    ("table_key", "bin100_table"), ("hires_scale", 0), ("lowres_scale", 4),
                    ("n_classes", 3), ("n_top_genes", 100), ("num_epochs", 1000),
                    ("device", "cpu"), ("n_jobs", -1), ("min_cells_per_type", 10),
                    ("min_cell_count", 1), ("min_segment_confidence", 0.0),
                    ("project_genes", True), ("plot_genes", None),
                ]}
            }
            proportions = config.get("cell_type_proportion")
            if proportions and i < len(proportions) and proportions[i]:
                params["cell_types_proportion"] = proportions[i]
                params["cell_types"] = set(proportions[i].keys())
            batch_params.append(params)
        
        run_batch(batch_params, max_workers=config.get("max_workers", 1))
    else:
        run_pipeline(
            sdata_path=config["sdata_path"],
            sc_path=config["sc_path"],
            output_dir=config["output_dir"],
            log_dir=config["log_dir"],
            library_id=config["library_id"],
            image_key=config["image_key"],
            table_key=config.get("table_key", "bin100_table"),
            hires_scale=config.get("hires_scale", 0),
            lowres_scale=config.get("lowres_scale", 4),
            n_classes=config.get("n_classes", 3),
            n_top_genes=config.get("n_top_genes", 100),
            num_epochs=config.get("num_epochs", 1000),
            device=config.get("device", "cpu"),
            n_jobs=config.get("n_jobs", -1),
            min_cells_per_type=config.get("min_cells_per_type", 10),
            min_cell_count=config.get("min_cell_count", 1),
            cell_types=set(config["cell_type_proportion"].keys()) if config.get("cell_type_proportion") else None,
            cell_types_proportion=config.get("cell_type_proportion"),
            min_segment_confidence=config.get("min_segment_confidence", 0.0),
            project_genes=config.get("project_genes", True),
            plot_genes=config.get("plot_genes"),
        )


def plot_main(args):
    """Run plotting."""
    from .pl import plot_tangram_proportion, run_batch as plot_run_batch

    setup_logging(verbose=args.verbose)
    input_path = pathlib.Path(args.config_or_h5ad)
    
    if input_path.suffix in (".yaml", ".yml"):
        config = load_config(args.config_or_h5ad)
        plot_run_batch(config)
    elif input_path.suffix == ".h5ad":
        if not args.output_dir or not args.library_id:
            print("ERROR: --output-dir and --library-id required for single file mode")
            sys.exit(1)
        plot_tangram_proportion(
            sp_h5ad_path=args.config_or_h5ad,
            output_dir=args.output_dir,
            library_id=args.library_id,
            dpi=args.dpi,
            cmap=args.cmap,
            perc=args.perc,
            suffix=args.suffix,
        )
    else:
        print(f"ERROR: Expected .yaml/.yml or .h5ad, got {input_path.suffix}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog="ezsp", description="Spatial transcriptomics analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # tangram subcommand
    tg_parser = subparsers.add_parser("tangram", help="Run Tangram deconvolution")
    tg_parser.add_argument("config", help="Path to YAML config file")
    tg_parser.set_defaults(func=tangram_main)
    
    # plot subcommand  
    pl_parser = subparsers.add_parser("plot", help="Plot Tangram results")
    pl_parser.add_argument("config_or_h5ad", help="YAML config or .h5ad file")
    pl_parser.add_argument("-o", "--output-dir", help="Output directory (single mode)")
    pl_parser.add_argument("-l", "--library-id", help="Library ID (single mode)")
    pl_parser.add_argument("--dpi", type=int, default=300)
    pl_parser.add_argument("--cmap", default="viridis")
    pl_parser.add_argument("--perc", type=float, default=2.0)
    pl_parser.add_argument("-s", "--suffix", default="")
    pl_parser.add_argument("-v", "--verbose", action="store_true")
    pl_parser.set_defaults(func=plot_main)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
