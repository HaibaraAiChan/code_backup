import dgl
import torch as th
from cpu_mem_usage import get_memory
import time
from ogb.nodeproppred import DglNodePropPredDataset


def ttt(tic, str1):
    toc = time.time()
    print(str1 + ' step Time(s): {:.4f}'.format(toc - tic))
    return toc


def load_ogbn_dataset(name, device, args):
    """
    Load dataset and move graph and features to device
    """
    '''if name not in ["ogbn-products", "ogbn-arxiv","ogbn-mag"]:
        raise RuntimeError("Dataset {} is not supported".format(name))'''
    if name not in ["ogbn-products", "ogbn-mag","ogbn-papers100M"]:
        raise RuntimeError("Dataset {} is not supported".format(name))
    dataset = DglNodePropPredDataset(name=name, root = args.root)
    splitted_idx = dataset.get_idx_split()
    print(name)

    # if name == "ogbn-products":
    #     train_nid = splitted_idx["train"]
    #     val_nid = splitted_idx["valid"]
    #     test_nid = splitted_idx["test"]
    #     g, labels = dataset[0]
    #     g.ndata["labels"] = labels
    #     g.ndata['feat'] = g.ndata['feat'].float()
    #     n_classes = dataset.num_classes
    #     labels = labels.squeeze()
    #     evaluator = get_ogb_evaluator(name)
    # elif name == "ogbn-mag":
    #     data = load_data(device, args)
    #     g, labels, n_classes, train_nid, val_nid, test_nid = data
    #     evaluator = get_ogb_evaluator(name)
    if name=="ogbn-papers100M":
        train_nid = splitted_idx["train"]
        val_nid = splitted_idx["valid"]
        test_nid = splitted_idx["test"]
        g, labels = dataset[0]        
        n_classes = dataset.num_classes        
        labels = labels.squeeze()
        evaluator = get_ogb_evaluator(name)        
        print(f"# Nodes: {g.number_of_nodes()}\n"
            f"# Edges: {g.number_of_edges()}\n"
            f"# Train: {len(train_nid)}\n"
            f"# Val: {len(val_nid)}\n"
            f"# Test: {len(test_nid)}\n"
            f"# Classes: {n_classes}\n")

        return g, labels, n_classes, train_nid, val_nid, test_nid, evaluator

# def load_ogbn_mag(name):
# 	from ogb.nodeproppred import DglNodePropPredDataset
# 	start = time.time()
# 	print('load', name)
# 	data = DglNodePropPredDataset(name=name)
# 	print('finish loading', name)

# 	hg_orig, labels = data[0]
# 	subgs = {}
# 	for etype in hg_orig.canonical_etypes:
# 		u, v = hg_orig.all_edges(etype=etype)
# 		subgs[etype] = (u, v)
# 		subgs[(etype[2], 'rev-' + etype[1], etype[0])] = (v, u)
# 	graph = dgl.heterograph(subgs)
# 	graph.nodes['paper'].data['feat'] = hg_orig.nodes['paper'].data['feat']
# 	split_idx = data.get_idx_split()
# 	train_idx = split_idx["train"]['paper']
# 	val_idx = split_idx["valid"]['paper']
# 	test_idx = split_idx["test"]['paper']
# 	paper_labels = labels['paper'].squeeze()
# 	print('labels')
# 	print(paper_labels)
# 	print('train_idx')
# 	print(train_idx)
# 	print('val_idx')
# 	print(val_idx)

# 	train_mask = th.zeros((graph.number_of_nodes('paper'),), dtype=th.bool)
# 	train_mask[train_idx] = True
# 	val_mask = th.zeros((graph.number_of_nodes('paper'),), dtype=th.bool)
# 	val_mask[val_idx] = True
# 	test_mask = th.zeros((graph.number_of_nodes('paper'),), dtype=th.bool)
# 	test_mask[test_idx] = True
# 	graph.nodes['paper'].data['train_mask'] = train_mask
# 	graph.nodes['paper'].data['val_mask'] = val_mask
# 	graph.nodes['paper'].data['test_mask'] = test_mask
# 	graph.nodes['paper'].data['labels'] = paper_labels
# 	print('graph')
# 	print(graph)
# 	print('load {} takes {:.3f} seconds'.format(name, time.time() - start))
# 	print('|V|={}, |E|={}'.format(graph.number_of_nodes(), graph.number_of_edges()))
# 	print('train: {}, valid: {}, test: {}'.format(th.sum(graph.nodes['paper'].data['train_mask']),
# 		th.sum(graph.nodes['paper'].data['val_mask']),
# 		th.sum(graph.nodes['paper'].data['test_mask'])))

# 	graph.nodes['paper'].ndata['features'] = graph.nodes['paper'].ndata['feat']
# 	graph.nodes['paper'].ndata['labels'] = paper_labels
# 	in_feats = graph.nodes['paper'].ndata['features'].shape[1]

# 	num_labels = len(th.unique(paper_labels[th.logical_not(th.isnan(paper_labels))]))

# 	print('finish constructing', name)

# 	return graph, num_labels


# def load_ogbn_mag(name):
#     from ogb.nodeproppred import DglNodePropPredDataset
#     print('load', name)
#     data = DglNodePropPredDataset(name=name)
#     print('finish loading', name)
#     splitted_idx = data.get_idx_split()

#     graph, labels = data[0]
#     labels = labels[:, 0]

#     graph.ndata['features'] = graph.ndata['feat']

#     graph.ndata['labels'] = labels

#     in_feats = graph.ndata['features'].shape[1]
#     num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

#     # Find the node IDs in the training, validation, and test set.
#     train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']

#     train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
#     train_mask[train_nid] = True
#     val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
#     val_mask[val_nid] = True
#     test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
#     test_mask[test_nid] = True
#     graph.ndata['train_mask'] = train_mask

#     graph.ndata['val_mask'] = val_mask

#     graph.ndata['test_mask'] = test_mask


#     print('finish constructing', name)


#     return graph, num_labels
    

#     # tic_step = time.time()
#     # get_memory("-" * 40 + "---------------------from ogb.nodeproppred import DglNodePropPredDataset***************************")
#     # print('load', name)
#     # data = DglNodePropPredDataset(name=name)
#     # t1 = ttt(tic_step, "-"*40+"---------------------data = DglNodePropPredDataset(name=name)***************************")
#     # # get_memory("-"*40+"---------------------data = DglNodePropPredDataset(name=name)***************************")
#     # print('finish loading', name)
#     # splitted_idx = data.get_idx_split()
#     # t2 = ttt(t1,"-" * 40 + "---------------------splitted_idx = data.get_idx_split()***************************")
#     # # get_memory("-" * 40 + "---------------------splitted_idx = data.get_idx_split()***************************")
#     # graph, labels = data[0]
#     # # get_memory("-" * 40 + "---------------------graph, labels = data[0]***************************")
#     # t3 = ttt(t2, "-" * 40 + "---------------------graph, labels = data[0]***************************")
#     # print(labels)
#     # print(data[0])
#     # print(graph)
#     # labels = labels[1][:, 0]
#     # # get_memory("-" * 40 + "---------------------labels = labels[:, 0]***************************")
#     # t4 = ttt(t3, "-" * 40 + "---------------------labels = labels[:, 0]***************************")

#     # graph.ndata['features'] = graph.ndata['feat']
#     # # get_memory("-" * 40 + "---------------------graph.ndata['features'] = graph.ndata['feat']***************************")
#     # t5 = ttt(t4, "-" * 40 + "---------------------graph.ndata['features'] = graph.ndata['feat']***************************")
#     # graph.ndata['labels'] = labels
#     # t6 = ttt(t5, "-" * 40 + "---------graph.ndata['labels'] = labels******************")
#     # in_feats = graph.ndata['features'].shape[1]
#     # num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

#     # # Find the node IDs in the training, validation, and test set.
#     # train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
#     # t7 = ttt(t6, "-" * 40 + "---------train_nid, val_nid, test_nid = splitted_idx******************")
#     # # get_memory(
# 	#     # "-" * 40 + "---------------------train_nid, val_nid, test_nid = splitted_idx***************************")
#     # train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
#     # train_mask[train_nid] = True
#     # val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
#     # val_mask[val_nid] = True
#     # test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
#     # test_mask[test_nid] = True
#     # graph.ndata['train_mask'] = train_mask
#     # graph.ndata['val_mask'] = val_mask
#     # graph.ndata['test_mask'] = test_mask
#     # t8 = ttt(t7, "-" * 40 + "---------end of load ogb******************")
#     # # get_memory(
# 	#     # "-" * 40 + "---------------------end of load ogb***************************")

#     # print('finish constructing', name)
#     # print('load ogb-products time total: '+ str(time.time()-tic_step))
#     # return graph, num_labels


def load_karate():
    from dgl.data import KarateClubDataset

    # load reddit data
    data = KarateClubDataset()
    g = data[0]
    print('karate data')
    # print(data[0].ndata)
    # print(data[0].edata)
    ndata=[]
    for nid in range(34):
        ndata.append((th.ones(4)*nid).tolist())
    ddd = {'feat': th.tensor(ndata)}
    g.ndata['features'] = ddd['feat']
    g.ndata['feat'] = ddd['feat']
    # print(data[0].ndata)
    g.ndata['labels'] = g.ndata['label']

    train = [True]*24 + [False]*10
    val = [False] * 24 + [True] * 5 + [False] * 5
    test = [False] * 24 + [False] * 5 + [True] * 5
    g.ndata['train_mask'] = th.tensor(train)
    g.ndata['val_mask'] = th.tensor(val)
    g.ndata['test_mask'] = th.tensor(test)

    return g, data.num_classes


def load_cora():
    from dgl.data import CoraGraphDataset

    # load reddit data
    data = CoraGraphDataset()
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    return g, data.num_classes


def load_reddit():
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True)
    g = data[0]
    g = dgl.remove_self_loop(g)
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    return g, data.num_classes

def load_ogb(name):

    tic_step = time.time()
    get_memory("-" * 40 + "---------------------from ogb.nodeproppred import DglNodePropPredDataset***************************")
    print('load', name)
    data = DglNodePropPredDataset(name=name)
    t1 = ttt(tic_step, "-"*40+"---------------------data = DglNodePropPredDataset(name=name)***************************")
    # get_memory("-"*40+"---------------------data = DglNodePropPredDataset(name=name)***************************")
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    t2 = ttt(t1,"-" * 40 + "---------------------splitted_idx = data.get_idx_split()***************************")
    # get_memory("-" * 40 + "---------------------splitted_idx = data.get_idx_split()***************************")
    graph, labels = data[0]

    graph = dgl.remove_self_loop(graph) #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    

    # get_memory("-" * 40 + "---------------------graph, labels = data[0]***************************")
    t3 = ttt(t2, "-" * 40 + "---------------------graph, labels = data[0]***************************")
    print(labels)
    print('graph after remove self connected edges')
    print(graph)
    labels = labels[:, 0]
    # get_memory("-" * 40 + "---------------------labels = labels[:, 0]***************************")
    t4 = ttt(t3, "-" * 40 + "---------------------labels = labels[:, 0]***************************")

    graph.ndata['features'] = graph.ndata['feat']
    # get_memory("-" * 40 + "---------------------graph.ndata['features'] = graph.ndata['feat']***************************")
    t5 = ttt(t4, "-" * 40 + "---------------------graph.ndata['features'] = graph.ndata['feat']***************************")
    graph.ndata['labels'] = labels
    graph.ndata['label'] = labels
    t6 = ttt(t5, "-" * 40 + "---------graph.ndata['labels'] = labels******************")
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    t7 = ttt(t6, "-" * 40 + "---------train_nid, val_nid, test_nid = splitted_idx******************")
    # get_memory(
	    # "-" * 40 + "---------------------train_nid, val_nid, test_nid = splitted_idx***************************")
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    t8 = ttt(t7, "-" * 40 + "---------end of load ogb******************")
    # get_memory(
	    # "-" * 40 + "---------------------end of load ogb***************************")

    print('finish constructing', name)
    print('load ogb-products time total: '+ str(time.time()-tic_step))
    return graph, num_labels

def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g
