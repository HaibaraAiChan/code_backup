import inspect
from torch.utils.data import DataLoader
from collections.abc import Mapping, Sequence
import backend as F

def _tensor_or_dict_to_numpy(ids):
	if isinstance(ids, Mapping):
		return {k: F.zerocopy_to_numpy(v) for k, v in ids.items()}
	else:
		return F.zerocopy_to_numpy(ids)



def partitioner(args, g, input_nodes, seed_nodes, exclude_eids=None):

	blocks = []
	# exclude_eids = (_tensor_or_dict_to_numpy(exclude_eids) if exclude_eids is not None else None)

	graph_device = g.device
	for block_id in reversed(range(args.num_layers)):
		seed_nodes_in = seed_nodes
		if isinstance(seed_nodes_in, dict):
			seed_nodes_in = {ntype: nodes.to(graph_device) for ntype, nodes in seed_nodes_in.items()}
		else:
			seed_nodes_in = seed_nodes_in.to(graph_device)
		frontier = args.sample_frontier(block_id, g, seed_nodes_in)

		# Removing edges from the frontier for link prediction training falls
		# into the category of frontier postprocessing
		if exclude_eids is not None:
			parent_eids = frontier.edata[EID]
			parent_eids_np = _tensor_or_dict_to_numpy(parent_eids)
			located_eids = _locate_eids_to_exclude(parent_eids_np, exclude_eids)
			if not isinstance(located_eids, Mapping):
				# (BarclayII) If frontier already has a EID field and located_eids is empty,
				# the returned graph will keep EID intact.  Otherwise, EID will change
				# to the mapping from the new graph to the old frontier.
				# So we need to test if located_eids is empty, and do the remapping ourselves.
				if len(located_eids) > 0:
					frontier = transform.remove_edges(
						frontier, located_eids, store_ids=True)
					frontier.edata[EID] = F.gather_row(parent_eids, frontier.edata[EID])
			else:
				# (BarclayII) remove_edges only accepts removing one type of edges,
				# so I need to keep track of the edge IDs left one by one.
				new_eids = parent_eids.copy()
				for k, v in located_eids.items():
					if len(v) > 0:
						frontier = transform.remove_edges(
							frontier, v, etype=k, store_ids=True)
						new_eids[k] = F.gather_row(parent_eids[k], frontier.edges[k].data[EID])
				frontier.edata[EID] = new_eids

		if args.output_device is not None:
			frontier = frontier.to(args.output_device)
			if isinstance(seed_nodes, dict):
				seed_nodes_out = {ntype: nodes.to(args.output_device) \
								  for ntype, nodes in seed_nodes.items()}
			else:
				seed_nodes_out = seed_nodes.to(args.output_device)
		else:
			seed_nodes_out = seed_nodes

		block = transform.to_block(frontier, seed_nodes_out)
		if args.return_eids:
			assign_block_eids(block, frontier)

		seed_nodes = {ntype: block.srcnodes[ntype].data[NID] for ntype in block.srctypes}
		blocks.insert(0, block)
		return blocks