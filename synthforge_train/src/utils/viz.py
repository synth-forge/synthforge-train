import torch
import numpy as np
import matplotlib.pyplot as plt
from ..data.d300w import D300WDataset
from ..data.lapa import LaPaDataset


class Visualizer:
    def __init__(self, n_cols, n_rows):
        self.n_cols = n_cols
        self.n_rows = n_rows

    def get_plot_kps_fn(self, imgs, tgt_kps, pred_kps, kp_vis):
        if kp_vis == 'scatter':
            def _plot_kps(ax, idx):
                img = imgs * 0.5 + 0.5
                ax.imshow(img[idx].cpu().permute(1, 2, 0))
                if tgt_kps is not None:
                    ax.scatter(*tgt_kps[idx].T.cpu(), s=1, c="b")
                if pred_kps is not None:
                    ax.scatter(*pred_kps[idx].T.cpu(), s=1, c="r")
                ax.axis("off")
        elif kp_vis == 'line':
            if tgt_kps is None:
                pass
            elif tgt_kps.size(-2) == 68:
                tgt_ds = D300WDataset
            elif tgt_kps.size(-2) == 106:
                tgt_ds = LaPaDataset
            else:
                raise ValueError(f'cant infer dataset from shape {tgt_kps.shape, tgt_kps.size(-2)}')
            
            if pred_kps is None:
                pass
            elif pred_kps.size(-2) == 68:
                pred_ds = D300WDataset
            elif pred_kps.size(-2) == 106:
                pred_ds = LaPaDataset
            else:
                raise ValueError(f'cant infer dataset from shape {pred_kps.shape, pred_kps.size(-2)}')

            def _plot_kps(ax, idx):
                img = imgs * 0.5 + 0.5
                ax.imshow(img[idx].cpu().permute(1, 2, 0))
                if tgt_kps is not None:
                    for is_closed, edge in tgt_ds.EDGE_INFO:
                        pts = tgt_kps[idx].T.cpu()[:, edge]
                        if is_closed:
                            pts = torch.cat([pts, pts[:, :1]], dim=-1)
                        ax.plot(*pts, c="b")
                ax.axis("off")
                if pred_kps is not None:
                    for is_closed, edge in pred_ds.EDGE_INFO:
                        pts = pred_kps[idx].T.cpu()[:, edge]
                        if is_closed:
                            pts = torch.cat([pts, pts[:, :1]], dim=-1)
                        ax.plot(*pts, c="r")
        else:
            raise ValueError(f'{kp_vis} not supported')

        return _plot_kps

    def get_plot_seg_fn(self, seg_map, img=None, w=0.25):
        def _plot_seg(ax, idx):
            seg = seg_map[idx]
            if img is not None:
                seg = w * (img[idx] * 0.5 + 0.5) + (1 - w) * seg
            ax.imshow(seg.cpu().permute(1, 2, 0))
            ax.axis("off")

        return _plot_seg

    def get_plot_depth_fn(self, depth_map):
        def _plot_seg(ax, idx):
            ax.imshow(depth_map[idx].cpu().squeeze())
            ax.axis("off")

        return _plot_seg

    def get_plot(
        self,
        imgs,
        px_tgts=None,
        px_pred=None,
        px_pred_la=None,
        tgt_seg_map=None,
        pred_seg_map=None,
        pred_seg_map_la=None,
        tgt_depth_map=None,
        pred_depth_map=None,
        pred_depth_map_la=None,
        kp_viz='scatter',
        extras=None,
    ):
        n_modalities = 1
        plotting_fns = [self.get_plot_kps_fn(imgs, px_tgts, px_pred, kp_viz)]
        if px_pred_la is not None:
            n_modalities += 1
            plotting_fns += [self.get_plot_kps_fn(imgs, px_tgts, px_pred_la, kp_viz)]

        if tgt_seg_map is not None:
            n_modalities += 1
            plotting_fns.append(self.get_plot_seg_fn(tgt_seg_map, img=imgs))
        if pred_seg_map is not None:
            n_modalities += 1
            plotting_fns.append(self.get_plot_seg_fn(pred_seg_map, img=imgs))
        if pred_seg_map_la is not None:
            n_modalities += 1
            plotting_fns.append(self.get_plot_seg_fn(pred_seg_map_la, img=imgs))
        if tgt_depth_map is not None:
            n_modalities += 1
            plotting_fns.append(self.get_plot_depth_fn(tgt_depth_map))
        if pred_depth_map is not None:
            n_modalities += 1
            plotting_fns.append(self.get_plot_depth_fn(pred_depth_map))
        if pred_depth_map_la is not None:
            n_modalities += 1
            plotting_fns.append(self.get_plot_depth_fn(pred_depth_map_la))
        
        if extras is not None:
            for k, plotting_fn in extras.items():
                n_modalities += 1
                plotting_fns.append(plotting_fn)
        
        fig, axs = plt.subplots(
            self.n_rows,
            self.n_cols * n_modalities,
            figsize=(2 * self.n_cols * n_modalities, 2 * self.n_rows),
            gridspec_kw={"wspace": 0, "hspace": 0},
        )
        for i, ax_row in enumerate(axs):
            for j, ax in enumerate(ax_row):
                idx = i * self.n_cols + (j // n_modalities)
                plot_fn = plotting_fns[j % n_modalities]
                plot_fn(ax, idx)
        plt.tight_layout(pad=0)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def close_plot(self):
        plt.close()
