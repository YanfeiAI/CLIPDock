import argparse
import os
import sys

from clipdock.utils.constant import VERSION


def build_parser():
    parser = argparse.ArgumentParser(
        description="CLIPDock: protein-ligand docking and virtual-screening rescoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--version", action="version",
                        version=f"%(prog)s {VERSION}")

    parser.add_argument("-r", "--receptor",
                        help="Receptor PDB file")
    ligand_group = parser.add_mutually_exclusive_group()
    ligand_group.add_argument("-l", "--ligand",
                              help="Ligand file (SDF, MOL2, etc.)")
    ligand_group.add_argument("-s", "--smi",
                              help="SMILES string of ligand")
    parser.add_argument("--ref",
                        help="Reference ligand used for pocket extraction and initial ligand centring (SDF, MOL2, etc.)")
    parser.add_argument("-o", "--output", default="output.sdf",
                        help="Output SDF file")
    parser.add_argument("--skip_extract_pocket", action="store_true",
                        help="Skip pocket extraction and use the full receptor for docking")
    parser.add_argument("--pad", type=int, default=8,
                        help="Width (Å) of the initial translation-sampling interval in each Cartesian dimension; initial translations are clipped to ±pad/2 around the centred starting pose. This does not constrain subsequent local optimization")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of worker processes")
    parser.add_argument("--restarts", type=int, default=32,
                        help="Number of restarts in the evolutionary algorithm")
    parser.add_argument("--pose_num", type=int, default=9,
                        help="Number of poses to save")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization of docking results")
    parser.add_argument("--score_only", action="store_true",
                        help="Score input poses without refinement")
    parser.add_argument("--minimize", action="store_true",
                        help="Perform local minimization only")
    parser.add_argument("--quiet", action="store_true",
                           help="Suppress output")
    parser.add_argument("--gscore", default=None,
                        help="Path to the exported G-score parameter CSV")

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    parser_vs = subparsers.add_parser(
        "vs",
        help="Batch docking and CLIPDock-score ranking",
        description="Dock a CSV library and rank compounds by CLIPDock score (lower is better)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_vs.add_argument("-i", "--input_file", required=True,
                           help="Input CSV: first column is name, second column is SMILES (first row skipped)")
    parser_vs.add_argument("-o", "--output_dir", required=True,
                           help="Output directory")
    parser_vs.add_argument("-r", "--receptor", required=True,
                           help="Receptor PDB file")
    parser_vs.add_argument("--ref", help="Reference ligand used for pocket extraction and initial placement of screened ligands")
    parser_vs.add_argument("--skip_extract_pocket", action="store_true",
                        help="Skip pocket extraction and use the receptor for docking")
    parser_vs.add_argument("--pad", type=int, default=8,
                        help="Width (Å) of the initial translation-sampling interval in each Cartesian dimension; initial translations are clipped to ±pad/2 around the centred starting pose. This does not constrain subsequent local optimization")
    parser_vs.add_argument("--workers", type=int, default=None,
                           help="Number of worker processes")
    parser_vs.add_argument("--restarts", type=int, default=32,
                           help="Number of restarts in the evolutionary algorithm")
    parser_vs.add_argument("--top_num", type=int, default=None,
                           help="Number of top molecules to export; omit to skip top-molecule export")
    parser_vs.add_argument("--top_pose_num", type=int, default=1,
                           help="Top poses per molecule")
    parser_vs.add_argument("--quiet", action="store_true",
                           help="Suppress output")
    parser_vs.add_argument("--gscore", default=argparse.SUPPRESS,
                           help="Path to the exported G-score parameter CSV")

    parser_vs_score = subparsers.add_parser(
        "vs-score",
        help="Rescore docked ligands with the graph-based VS-score model",
        description="Rescore docked ligands with the graph-based VS-score model (higher is better)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_vs_score.add_argument("-p", "--pocket", required=True,
                                 help="Receptor pocket PDB file")
    score_input = parser_vs_score.add_mutually_exclusive_group(required=True)
    score_input.add_argument("-l", "--ligands",
                             help="SDF file containing one or more docked ligands")
    score_input.add_argument("--lig_dir",
                             help="Directory containing docked ligand SDF files")
    parser_vs_score.add_argument("-m", "--model", default=None,
                                 help="Path to a custom VS-score checkpoint")
    parser_vs_score.add_argument("-o", "--output", required=True,
                                 help="Output filename prefix; .csv is appended automatically")
    parser_vs_score.add_argument("--device", default="0",
                                 help="CUDA device index, or -1 for CPU inference")
    parser_vs_score.add_argument("--batch_size", type=int, default=12,
                                 help="Inference batch size")
    parser_vs_score.add_argument("--num_workers", type=int, default=12,
                                 help="DataLoader worker processes")
    parser_vs_score.add_argument("--top_dir", default=None,
                                 help="Output directory for selected top molecules")
    parser_vs_score.add_argument("--top_num", type=int, default=None,
                                 help="Select and export this many top-scoring molecules")
    parser_vs_score.add_argument("--cluster", type=float, default=0.0,
                                 help="Butina clustering threshold; 0 disables clustering")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        return

    if args.command == "vs":
        if not os.path.isfile(args.input_file):
            parser.error(f"input CSV does not exist: {args.input_file}")
        if not os.path.isfile(args.receptor):
            parser.error(f"receptor file does not exist: {args.receptor}")
        if args.ref is not None and not os.path.isfile(args.ref):
            parser.error(f"reference ligand does not exist: {args.ref}")
        if args.ref is None and not args.skip_extract_pocket:
            parser.error("clipdock vs requires --ref unless --skip_extract_pocket is used")
        if args.gscore is not None and not os.path.isfile(args.gscore):
            parser.error(f"G-score parameter file does not exist: {args.gscore}")
        if args.gscore is not None:
            args.gscore = os.path.abspath(args.gscore)
        from clipdock.vs import main as vs_main
        vs_main(args)
    elif args.command == "vs-score":
        if not os.path.isfile(args.pocket):
            parser.error(f"pocket file does not exist: {args.pocket}")
        if args.ligands is not None and not os.path.isfile(args.ligands):
            parser.error(f"ligand SDF does not exist: {args.ligands}")
        if args.lig_dir is not None and not os.path.isdir(args.lig_dir):
            parser.error(f"ligand directory does not exist: {args.lig_dir}")
        if args.model is not None and not os.path.isfile(args.model):
            parser.error(f"model checkpoint does not exist: {args.model}")
        from model.graph_score import run
        run(args)
    else:
        if args.receptor is None:
            parser.error("docking requires -r/--receptor")
        if args.ligand is None and args.smi is None:
            parser.error("docking requires exactly one of -l/--ligand or -s/--smi")
        if not os.path.isfile(args.receptor):
            parser.error(f"receptor file does not exist: {args.receptor}")
        if args.ligand is not None and not os.path.isfile(args.ligand):
            parser.error(f"ligand file does not exist: {args.ligand}")
        if args.ref is not None and not os.path.isfile(args.ref):
            parser.error(f"reference ligand does not exist: {args.ref}")
        if args.gscore is not None and not os.path.isfile(args.gscore):
            parser.error(f"G-score parameter file does not exist: {args.gscore}")
        if args.gscore is not None:
            args.gscore = os.path.abspath(args.gscore)
        if args.smi is not None and args.ref is None and not args.skip_extract_pocket:
            parser.error("SMILES docking requires --ref unless --skip_extract_pocket is used")
        if args.score_only and args.ligand is None:
            parser.error("--score_only requires -l/--ligand with existing coordinates; SMILES input is not supported")
        if args.score_only and args.minimize:
            parser.error("--score_only and --minimize cannot be used together")
        from clipdock.dock import main as dock_main
        dock_main(args)


if __name__ == "__main__":
    main()
