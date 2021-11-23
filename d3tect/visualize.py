
import pandas as pd
from logging import getLogger, debug, info, warn, error, DEBUG, INFO
import igraph

import numpy as np
import seaborn as sns
from math import sqrt, ceil
from d3tect.worker import worker
from d3tect.model import Technique, Datasource
from operator import attrgetter
import matplotlib.pyplot as plt
import re

class Visualize:
    @staticmethod
    def plot_heatmap__test(worker1, sort_ds='no_of_techniques', sorting='rank_no_examples_grp'):
        matrix_xy = ceil(sqrt(len(worker1.t_list))) #get sqrt from count of techniques, round up; we need this for the size of our heatmap
        matrix_size = matrix_xy * matrix_xy
        print(matrix_size)

        ### BUILD HEATMAPS
        # create list of techniques sorted by kill chain, and then sorted by

        # create list sorted by most often used technique
        text = []
        data = []
        for t in sorted(worker1.t_dict.values(), key=attrgetter(Technique.ranks_t[2]), reverse=False):
            text.append(t.id)
            data.append(float(t.no_examples_weighted))

        while len(text) < matrix_size:
            text.append('filler')
            data.append(0)
            #data = t.rank_no_examples_weighted

        text = np.asarray(text).reshape(matrix_xy,matrix_xy)
        data = np.asarray(data).reshape(matrix_xy,matrix_xy)

        # create list sorted by hard to detect most often used technique
        # hard to detect means less datasources and high usability
        # or that are only detected by datasources that do not detect a lot of techniques and are therefore "expensive"

        #https://likegeeks.com/seaborn-heatmap-tutorial/

        #data = np.random.rand(15,15)
        #text = np.asarray([['a', 'b', 'c', 'd', 'e', 'f'], ['g', 'h', 'i', 'j', 'k', 'l'], ['m', 'n', 'o', 'p', 'q', 'r'], ['s', 't', 'u', 'v', 'w', 'x']])
        labels = (np.asarray(["{0}\n{1}".format(text,data) for text, data in zip(text.flatten(), data.flatten())])).reshape(matrix_xy, matrix_xy)

        ax = sns.heatmap(data, xticklabels=False, yticklabels=False, cmap="PuBu", annot=labels, fmt='')
        ##ax.set_aspect("equal")
        sns.set(font_scale=1)
        plt.show()

    @staticmethod
    def get_plot_scatter_density(ds_dict, t_dict, levels=0, lines=False, y='Max DS Value of Technique (log)', opacity=1.0, text_for_ids=[], ds_rank=4, t_rank=3):
        df = {'id':[], 'name':[], 'Technique Rank':[], 'tech_val':[], 'max_ds_val_of_tech':[], 'Max DS Value of Technique (log)':[], 'Path excludes':[], 'min_ds_rank':[], 'text_for_ids':[]}

        def extend_df(ds_dict, t_dict, path_ex):
            worker.calc_t_ds_val(ds_dict, t_dict, ds_rank)

            df['id'].extend([t.id for t in t_dict.values()])
            df['name'].extend([t.name for t in t_dict.values()])
            df['tech_val'].extend([t.get_metric(t_rank) for t in t_dict.values()])
            df['Technique Rank'].extend([t.get_rank(t_rank) for t in t_dict.values()])
            df['max_ds_val_of_tech'].extend([t.datasource_val['max_ds_val'] for t in t_dict.values()])
            df['Path excludes'].extend([path_ex for t in t_dict.values()])
            df['min_ds_rank'].extend([t.datasource_val['min_ds_rank'] for t in t_dict.values()])
            df['text_for_ids'].extend([f'<b>{t.id}</b>' if t.id in text_for_ids else None for t in t_dict.values()])


        if levels:
            excludes = []
            path_list = []
            path_excludes = []

            for level in range(levels):
                new_path = worker.get_shortest_path_for_detection(sort_type_ds=ds_rank, ref_ds_dict=ds_dict,
                                                                     ref_t_dict=t_dict, excluded_ds=excludes)
                path_excludes.append(excludes.copy())
                path_list.append(new_path)
                excludes.append(new_path['ds'].popitem(last=False)[0])

            for path in path_list:
                ds_dict = path['ds']
                t_dict = path['t']
                excl = path_excludes.pop(0)
                if not excl:
                    path_ex = "None"
                else:
                    path_ex = '<br>'.join(excl)

                extend_df(ds_dict, t_dict, path_ex)
        else:
            extend_df(ds_dict, t_dict, 'None')

        df['Max DS Value of Technique (log)'] = [np.log(dsval) if dsval else 0 for dsval in df['max_ds_val_of_tech']]
        #print(df)

        if(levels):
            color='Path excludes'
        else:
            color=None

        #y='max_ds_val_of_tech'
        import plotly.express as px
        fig = px.scatter(df, x='Technique Rank', y=y, opacity=opacity, text='text_for_ids', #log_y=True,
                         hover_data=['id', 'name', 'tech_val', 'max_ds_val_of_tech', 'Path excludes'], color=color)
        #fig.update_traces(textposition='top center', fill='tonexty')
        fig.for_each_trace(lambda t: t.update(textfont_color=t.marker.color, textposition='top center'))
        #fig.for_each_trace(lambda t: print(t))
        #fig.for_each_trace(lambda t: t.update(bordercolor="#c7c7c7", borderwidth=2, borderpad=4, bgcolor="#ff7f0e"))
        if lines and not levels:
            far_right = len(df['Technique Rank'])+20
            far_left = -35
            ds_ranked = dict()
            for ds in ds_dict.values():
                if (ds.metric['total_no_examples_weighted'] > 0):
                    ds_ranked[ds.rank['total_no_examples_weighted']] = [ds.name, np.log(ds.metric['total_no_examples_weighted'])]
            max_ranked_ds = len(ds_ranked)
            def addano(top_x):
                fig.add_annotation(x=far_left, y=ds_ranked[top_x][1],
                                   text=f"Top {top_x} DS", xanchor="left", font=dict(color="grey"),
                                   showarrow=False, yshift=10)
                fig.add_shape(type="line",
                              x0=far_left, y0=ds_ranked[top_x][1], x1=far_right, y1=ds_ranked[top_x][1],
                              line=dict(color="grey", width=1, dash="dash"))

            addano(10)
            addano(20)
            addano(30)
            addano(50)
            addano(70)

            #fig.add_shape(type="text", x0=far_left,  -0.6, r'$\sum_{i=0}^\infty x_i$', fontsize=20)
            """fig.add_shape(type="line",
                          x0=far_left, y0=ds_ranked[40][1], x1=far_right, y1=ds_ranked[40][1],
                          line=dict(color="black", width=1, dash="solid"))
            fig.add_shape(type="line",
                          x0=far_left, y0=ds_ranked[50][1], x1=far_right, y1=ds_ranked[50][1],
                          line=dict(color="black", width=1, dash="solid"))
            fig.add_shape(type="line",
                          x0=far_left, y0=ds_ranked[60][1], x1=far_right, y1=ds_ranked[60][1],
                          line=dict(color="black", width=1, dash="solid"))"""

        df = pd.DataFrame(data=df)
        #fig['layout']['xaxis']['autorange'] = "reversed"
        fig.show()



    @staticmethod
    def get_levels_shortest_path_for_detection(sort_type_ds=4, sort_type_ds_vertic=4, ref_ds_dict=None, ref_t_dict=None, name_ref_ds_dict=None, worker=None,
                                                 excluded_ds=[], max_levels=3, fig_height='900px'):
        if not ref_ds_dict: return
        if not name_ref_ds_dict: name_ref_ds_dict=ref_ds_dict
        if not ref_t_dict: return
        total_detectable_t = len(worker.get_detectable_techniques(ref_t_dict))
        sum_weight_total_detectable_t = sum([ref_t_dict[t].metric['no_examples_weighted'] for t in worker.get_detectable_techniques(ref_t_dict)])

        class Node:
            i = []
            max_nodes = 10

            def __init__(self, parent, function, excludes, includes, sort_type_ds, ref_ds_dict, ref_t_dict):
                self.function = function
                self.excludes = excludes
                self.includes = includes
                self.sort_type_ds = sort_type_ds
                self.ref_ds_dict = ref_ds_dict
                self.ref_t_dict = ref_t_dict
                self.parent = parent
                self.id = self.get_uname()
                self.left = None
                self.right = None
                self.ds_path = function.get_shortest_path_for_detection(sort_type_ds=sort_type_ds, ref_ds_dict=ref_ds_dict,
                                                     ref_t_dict=ref_t_dict, excluded_ds=excludes)
                self.ds_path_obj_list = self.ds_path['ds']
                self.remove_includes()

                self.ds_name, self.ds_obj = self.ds_path_obj_list.popitem(last=False)
                self.missing = self.ds_path['missing']

            def remove_includes(self):
                for key in self.includes:
                    del(self.ds_path_obj_list[key.name])

            def get_left(self):
                if not self.left:
                    #new_excludes = [self.ds_name] + self.excludes
                    new_includes = [self.ds_obj] + self.includes
                    self.left = Node(self, self.function, self.excludes, new_includes, self.sort_type_ds, self.ref_ds_dict, self.ref_t_dict)
                    debug(f'We extended {self.id} to left {self.left.id} with excludes {self.excludes} \n{self.ds_path_obj_list}\n  new chain\n {self.left.ds_path_obj_list}\n')
                return self.left

            def get_right(self):
                if not self.right:
                    new_excludes = [self.ds_name] + self.excludes
                    self.right = Node(self, self.function, new_excludes, self.includes, self.sort_type_ds, self.ref_ds_dict, self.ref_t_dict)
                    debug(f'We extended  {self.id} to right {self.right.id} with new excludes {new_excludes} \n{self.ds_path_obj_list}\n new chain\n {self.right.ds_path_obj_list}\n')
                return self.right

            def get_uname(self):
                i = 0
                while(i in Node.i):
                    i+=1
                Node.i.append(i)
                return(i)

            def get_len_detections_in_path(self):
                return sum([len(ds.techniques_in_detection) for ds in self.includes])

            def get_val_detections_in_path(self, sort_type_ds):
                return sum([ds.get_metric(sort_type_ds) for ds in self.includes])

            def get_val_missing_in_path(self, sort_type_t):
                return sum([self.ref_t_dict[t].get_metric(sort_type_t) for t in self.missing])

            def get_detection_value(self, sort_type_ds):
                return self.ds_obj.get_metric(sort_type_ds)

            def get_detections(self):
                return len(self.ds_obj.techniques_in_detection)

            def get_rank_top_ds(self, sort_type_ds):
                return self.ds_obj.get_rank(sort_type_ds)

        no_vertices = pow(2, max_levels + 1) - 1
        pl = igraph.Plot()
        g = igraph.Graph()
        g.add_vertices(no_vertices)

        root_node = Node(None, worker, [], [], sort_type_ds, ref_ds_dict, ref_t_dict)

        all_nodes = []
        nodes = [[] for x in range(0, max_levels+1)]
        edge_text = []
        nodes[0] = [root_node]
        all_nodes += nodes[0]
        for level in range (0, max_levels):
            for node in nodes[level]:
                left = node.get_left()
                right = node.get_right()
                g.add_edges([(left.parent.id, left.id), (right.parent.id, right.id)])

                l_ds_id = f'{name_ref_ds_dict[left.includes[0].name].get_rank(sort_type_ds_vertic)}'
                if level > 2: # level2
                    l_ds_name = l_ds_id
                else:
                    l_ds_name = getattr(name_ref_ds_dict[left.includes[0].name], "name")
                    l_ds_name = f'{l_ds_id}: {l_ds_name.split(":")[0]}:<br>{l_ds_name.split(":")[1]}'
                right_text = f'{__class__.calc_perc(len(right.missing), total_detectable_t)}'
                edge_text += [l_ds_name, right_text]

                nodes[level+1].append(left)
                nodes[level+1].append(right)
            all_nodes.extend(nodes[level+1])

        layout = g.layout("rt", root=(0, 0))
        #pl.add(g, layout=layout);
        #pl._windows_hacks = True;
        #pl.show();

        #nr_vertices = 25
        vertices_label = [f'Name: {x.ds_obj.name}<br>' \
                          f'New detections: {x.get_detections()}<br>' \
                          f'New value: {x.get_detection_value(sort_type_ds)}<br>' \
                          f'Detected in Path: {x.get_len_detections_in_path()} this is {__class__.calc_perc(x.get_len_detections_in_path(), total_detectable_t)}<br>' \
                          f'Detected val in Path: {x.get_val_detections_in_path(sort_type_ds):.2f} this is {__class__.calc_perc(x.get_val_detections_in_path(sort_type_ds), sum_weight_total_detectable_t)}<br>' \
                          f'Undetectable in Path: {len(x.missing)} this is {__class__.calc_perc(len(x.missing), total_detectable_t)}<br>' \
                          f'Undetectable val in Path: {x.get_val_missing_in_path(sort_type_ds-1):.2f} this is {__class__.calc_perc(x.get_val_missing_in_path(sort_type_ds-1), sum_weight_total_detectable_t)}<br>' \
                          f'Rank within Path: {x.get_rank_top_ds(sort_type_ds)}' for x in all_nodes]
        #vertices_label_rank = [getattr(name_ref_ds_dict[x.ds_obj.name], Datasource.ranks_t[sort_type_ds_vertic]) for x in all_nodes]
        vertices_label_rank = [__class__.calc_perc(x.get_len_detections_in_path(), total_detectable_t) for x
                               in all_nodes]

        #G = Graph.Tree(nr_vertices, 2)  # 2 stands for children number
        #lay = G.layout('rt')

        position = {k: layout[k] for k in range(no_vertices)}
        Y = [layout[k][1] for k in range(no_vertices)]
        M = max(Y)

        es = igraph.EdgeSeq(g)  # sequence of edges
        E = [e.tuple for e in g.es]  # list of edges

        L = len(position)
        Xn = [position[k][0] for k in range(L)]
        Yn = [2 * M - position[k][1] for k in range(L)]
        Xe = []
        Ye = []
        XeA = []
        YeA = []
        for edge in E:
            X = [position[edge[0]][0], position[edge[1]][0], None]
            Y = [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]
            Xe += X
            Ye += Y
            XeA.append((X[0] + X[1])/2)
            YeA.append((Y[0] + Y[1])/2)

        def make_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
            L = len(pos)
            if len(text) != L:
                raise ValueError(f'The lists pos and text must have the same len {len(text)} {L} ')
            annotations = []
            for k in range(L):
                annotations.append(
                    dict(
                        text=vertices_label_rank[k],  # or replace labels with a different list for the text within the circle
                        x=pos[k][0], y=2 * M - position[k][1],
                        xref='x1', yref='y1',
                        font=dict(color=font_color, size=font_size),
                        showarrow=False)
                )
            return annotations

        import plotly.graph_objects as go
        fig = go.Figure()
        axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    )

        fig.add_trace(go.Scatter(x=Xe,
                                 y=Ye,
                                 mode='lines',
                                 line=dict(color='rgb(210,210,210)', width=3),
                                 ))


        fig.add_trace(go.Scatter(x=Xn,
                                 y=Yn,
                                 mode='markers',
                                 #name='bla',
                                 marker=dict(symbol='circle-dot',
                                             size=34,
                                             color='rgb(50,50,50)', #'#6175c1',  # '#DB4551',
                                             line=dict(color='rgb(50,50,50)', width=2)
                                             ),
                                 text=vertices_label,
                                 opacity=1
                                 ))


        fig.update_layout(#title='Tree with Reingold-Tilford Layout',
                          annotations=make_annotations(position, vertices_label),
                          font_size=12,
                          showlegend=False,
                          xaxis=axis,
                          yaxis=axis,
                          margin=dict(l=40, r=40, b=85, t=100),
                          hovermode='closest',
                          #plot_bgcolor='rgb(248,248,248)',
                          paper_bgcolor = 'rgba(0,0,0,0)',
                          plot_bgcolor = 'rgba(0,0,0,0)',
                          height = fig_height
                          )


        for (x, y, text) in zip(XeA, YeA, edge_text):
            fig.add_annotation(x=x, y=y, text=text, #https://plotly.com/python/text-and-annotations/#adding-annotations-with-xref-and-yref-as-paper
                               bordercolor="#c7c7c7",
                               borderwidth=1,
                               borderpad=0,
                               bgcolor="white",
                               opacity=0.8,
                               showarrow=False, arrowhead=1)
        debug(f'Total detectable {total_detectable_t}')
        debug(f'Total sum of detectable {sum_weight_total_detectable_t}')
        fig.show()

    @staticmethod
    def calc_perc(part, whole):
        percentage = 100 * float(part)/float(whole)
        return f'{percentage:.1f}%'

class Visualize_old:
    def __init__(self, worker):
        self.worker = worker

    def build_even_matrix(self, text, data, file='', pr_label='Full', cmap="rocket_r", cbar=True):
        matrix_xy = ceil(sqrt(
            len(self.worker.t_dict)))  # get sqrt from count of techniques, round up; we need this for the size of our heatmap
        matrix_size = matrix_xy * matrix_xy
        # print(matrix_size)

        while len(text) < matrix_size:
            text.append('-')
            data.append(np.nan)

        text = np.asarray(text).reshape(matrix_xy, matrix_xy)
        data = np.asarray(data).reshape(matrix_xy, matrix_xy)

        if pr_label == 'Full':
            labels = (
                np.asarray(
                    ["{0}\n{1}".format(text, data) for text, data in zip(text.flatten(), data.flatten())])).reshape(
                matrix_xy, matrix_xy)
        elif pr_label == 'Simple':
            labels = text
        elif pr_label == 'None':
            labels = None

        # cmap="PuBu_r"
        ax = sns.heatmap(data, xticklabels=False, yticklabels=False, cmap=cmap, annot=labels, fmt='', cbar=cbar)
        plt.tight_layout()
        # plt.subplots_adjust(top=0.95)

        sns.set(font_scale=1)

        if file == '':
            plt.show()
        else:
            # sns_plot = sns.pairplot(ax, hue='species', size=2.5)
            plt.savefig(file)
        plt.clf()

    ### BUILD HEATMAPS
    # TODO create list of techniques sorted by kill chain, and then sorted by ???

    # TODO create list sorted by hard to detect most often used technique
    # hard to detect means less datasources and high usability
    # or ds that are only detected by datasources that do not detect a lot of techniques and are therefore "expensive"

    # create list sorted by most often used technique3
    def even_matrix_1(self, heatmap_t_attr='no_of_datasources'):
        text = []
        data = []

        for t in sorted(self.worker.t_dict.values(), key=lambda x: x.get_rank(2), reverse=False):
            text.append(t.id)
            data.append(t.metric[heatmap_t_attr])  # BUG EMPTY DATASOURCES COUNT AS ONE

        self.build_even_matrix(text, data)

    # create list sorted by most often used technique3
    def even_matrix_max_dscore(self, heatmap_t_attr='total_no_examples_weighted'):
        text = []
        data = []
        # (ref_ds_dict.values(), key= lambda x: x.get_rank(sort_type_ds), reverse = False):
        for t in sorted(self.worker.t_dict.values(), key=lambda x: x.get_rank(2), reverse=False):
            text.append(t.id)
            data.append(round(t.get_max_ds_score(key=heatmap_t_attr), 0))

        self.build_even_matrix(text, data)

    # create list sorted by most often used technique3 and plot binary repr techniques detected by datasource
    def even_matrix_bin_coverage(self, datasource, to_file=False):
        text = []
        data = []

        t_from_ds = [t.id for t in self.worker.ds_dict[datasource].techniques]
        #print(t_from_ds)

        for t in sorted(self.worker.t_dict.values(), key=lambda x: x.get_rank(2), reverse=False):
            text.append(t.id)
            # if(t.datasources)
            if t.id in t_from_ds:
                data.append(1)
            else:
                data.append(0)

        file = '' if not to_file else 'exported_stats/binary_ds/{}.png'.format(
            re.sub('[^a-zA-Z0-9 \n\.]', '', datasource))
        self.build_even_matrix(text, data, pr_label='None', file=file, cbar=False)

    def even_matrix_bin_coverage_all(self):
        print(self.worker.ds_dict)
        for ds in self.worker.ds_dict.values():
            self.even_matrix_bin_coverage(ds.name, to_file=True)

    def attack_matrix_1(self, heatmap_t_attr='no_examples_grp'):
        # get first technique object to access static tactic dict and build matrix
        max_size = 0
        for tactic, obj in (next(iter(self.worker.t_dict.values())).tactic_dict).items():
            new_size = len(obj)
            if (new_size > max_size):
                max_size = new_size
                print('new max_size {} from {}'.format(max_size, tactic))

        arr_x = len((next(iter(self.worker.t_dict.values())).tactic_dict).keys())
        arr_y = max_size

        # init with max size
        data = np.empty((arr_y, arr_x))
        data[:] = np.NaN

        # inits as string with len10
        text = np.empty((arr_y, arr_x), dtype='U10')

        tacti = 0
        for tactic, obj in (next(iter(self.worker.t_dict.values())).tactic_dict).items():
            print(tactic)
            print(len(obj))

            techi = 0
            for o in obj:
                data[techi, tacti] = getattr(o, heatmap_t_attr)
                text[techi, tacti] = '{}: {}'.format(o.id, getattr(o, heatmap_t_attr))
                techi += 1
            tacti += 1

        # cmap="PuBu_r"
        ax = sns.heatmap(data, xticklabels=False, yticklabels=False, cmap="rocket_r", annot=text, fmt='')
        ##ax.set_aspect("equal")
        sns.set(font_scale=1)
        plt.show()