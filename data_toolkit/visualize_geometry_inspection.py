"""Plot orthographic edge diagnostics from inspect_geometry.py JSON reports."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
import numpy as np


def main(args):
    reports = [json.loads(path.read_text()) for path in args.reports]
    fig, axes = plt.subplots(len(reports), 3, figsize=(16, 6*len(reports)), squeeze=False)
    fig.patch.set_facecolor('#f5f7fa')
    for row, report in enumerate(reports):
        objects = report['objects']
        pre = sum(o['before_weld']['multi_face_edges'] for o in objects)
        post = sum(o['after_weld']['multi_face_edges'] for o in objects)
        boundary = sum(o['after_weld']['boundary_edges'] for o in objects)
        for col, (dims, title) in enumerate([([0,2], 'Front  /  XZ'), ([0,1], 'Top  /  XY'), ([1,2], 'Side  /  YZ')]):
            ax = axes[row,col]
            ax.set_facecolor('#f5f7fa')
            for obj in objects:
                vertices = np.asarray(obj['preview_vertices'])
                triangles = np.asarray(obj['preview_triangles'], dtype=int)
                if len(triangles):
                    polys = vertices[triangles][:,:,dims]
                    ax.add_collection(PolyCollection(polys, facecolors='#8b9eb3',
                                                     edgecolors='#6b8098', linewidths=.12, alpha=.12))
                for key, color, width, alpha in [('boundary_segments','#19aeca',.4,.6),
                                                 ('multi_face_segments','#f23b35',1.5,1)]:
                    segments = np.asarray(obj['after_weld'][key])
                    if len(segments):
                        ax.add_collection(LineCollection(segments[:,:,dims], colors=color,
                                                         linewidths=width, alpha=alpha))
            ax.set(xlim=(-.56,.56), ylim=(-.56,.56), aspect='equal')
            ax.set_title(title, loc='left', fontsize=13, color='#364861', pad=8)
            ax.axis('off')
            if col == 0:
                ax.text(0,1.17,report['asset'][:8]+'  |  edge inspection', transform=ax.transAxes,
                        fontsize=18, weight='bold', color='#172a45')
                ax.text(0,1.09,f'>2-face edges: {pre} before weld → {post} after weld\n'
                        f'Allowed open boundaries after cleanup: {boundary:,}', transform=ax.transAxes,
                        fontsize=10, color='#53657c', linespacing=1.5)
    fig.legend(handles=[Line2D([0],[0],color='#f23b35',lw=2,label='>2 adjacent faces  /  preserved and tagged'),
                        Line2D([0],[0],color='#19aeca',lw=2,label='1 adjacent face  /  allowed open boundary'),
                        Line2D([0],[0],color='#8b9eb3',lw=6,alpha=.4,label='Geometry projection')],
               loc='lower center',ncol=3,frameon=False,fontsize=10,bbox_to_anchor=(.5,.012))
    fig.subplots_adjust(left=.04,right=.98,top=.90,bottom=.075,wspace=.12,hspace=.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output,dpi=180,facecolor=fig.get_facecolor())
    print(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    main(parser.parse_args())
