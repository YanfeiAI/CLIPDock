import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model.graph_model import CLIPDockVS
from model.graph_data import MoleculeDataModule
from model.graph_utils import get_cosine_schedule_with_warmup, mdn_loss_fn, disable_rdkit_logging

disable_rdkit_logging()

torch.set_float32_matmul_precision('high')

USE_AMP_GLOBAL = False
DIST_THRESHOLD = 8.0

class VSTrainer(pl.LightningModule):
    def __init__(self, prot_node_in=80, prot_edge_in=7, lig_node_in=48, lig_edge_in=7,
                 hidden_dim=256, num_layers=9, num_gaussians=16, lr=1e-3, dist_threshold=8.0,
                 dropout=0.1, num_heads=8):
        super().__init__()
        self.save_hyperparameters()
        self.model = CLIPDockVS(prot_node_in, prot_edge_in, lig_node_in, lig_edge_in,
                                hidden_dim, num_layers, num_gaussians, dropout, dist_threshold,
                                num_heads)
        self.lr = lr
        self.dist_threshold = dist_threshold

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        prot, lig, target_dist = batch
        pi, mu, sigma, pair_counts = self.model(
            prot['x'], prot['edge_index'], prot['edge_attr'], prot['batch'],
            lig['x'], lig['edge_index'], lig['edge_attr'], lig['batch']
        )
        loss = mdn_loss_fn(pi, mu, sigma, target_dist, dist_threshold=self.dist_threshold)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, batch_size=len(pair_counts))
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None
        prot, lig, target_dist = batch
        pi, mu, sigma, pair_counts = self.model(
            prot['x'], prot['edge_index'], prot['edge_attr'], prot['batch'],
            lig['x'], lig['edge_index'], lig['edge_attr'], lig['batch']
        )
        loss = mdn_loss_fn(pi, mu, sigma, target_dist, dist_threshold=self.dist_threshold)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, batch_size=len(pair_counts))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        total_steps = self.trainer.estimated_stepping_batches

        scheduler = get_cosine_schedule_with_warmup(optimizer, total_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }

def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the graph-based CLIPDock VS-score model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training and validation batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial AdamW learning rate')
    parser.add_argument('--epochs', type=int, default=600,
                        help='Number of training epochs')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device index, or -1 for CPU training')
    parser.add_argument('--data_suffix', type=str, required=True,
                        help='Suffix shared with the preceding G-score docking outputs, for example r8p32')
    parser.add_argument('--train_split', type=str, default='pdbpl_train',
                        help='Training split passed to MoleculeDataModule')
    parser.add_argument('--val_split', type=str, default='pdbpl_val',
                        help='Validation split passed to MoleculeDataModule')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Worker processes used for graph preprocessing and loading')
    parser.add_argument('--output_dir', type=Path, default=Path('data/training/vs_score'),
                        help='Training output directory')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_module = MoleculeDataModule(train_split=args.train_split,
                                     val_split=args.val_split,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     pocket_cutoff=DIST_THRESHOLD,
                                     data_suffix=args.data_suffix)

    model = VSTrainer(lr=args.lr, dist_threshold=DIST_THRESHOLD)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        save_top_k=0,
        save_weights_only=True,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        patience=50,
        mode='min',
        verbose=True
    )

    use_gpu = args.device >= 0
    trainer = pl.Trainer(max_epochs=args.epochs,
                         accelerator='gpu' if use_gpu else 'cpu',
                         devices=[args.device] if use_gpu else 1,
                         precision='16-mixed' if USE_AMP_GLOBAL and use_gpu else '32',
                         gradient_clip_val=1.0,
                         callbacks=[checkpoint_callback, early_stop_callback],
                         logger=False,
                         default_root_dir=output_dir)
    trainer.fit(model, datamodule=data_module)
    last_model_path = checkpoint_callback.last_model_path
    if not last_model_path:
        raise RuntimeError('VS-score checkpoint was not saved')
    checkpoint_path = Path(last_model_path)
    if not checkpoint_path.is_file():
        raise RuntimeError(f'VS-score checkpoint was not saved: {checkpoint_path}')
    print(f'VS-score checkpoint saved to {checkpoint_path}')

if __name__ == '__main__':
    main()
