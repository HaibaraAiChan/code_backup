from dgl import DGLHeteroGraph
class DGLBlock(DGLHeteroGraph):
    """Subclass that signifies the graph is a block created from
    :func:`dgl.to_block`.
    """
    # (BarclayII) I'm making a subclass because I don't want to make another version of
    # serialization that contains the is_block flag.
    is_block = True

    def __repr__(self):
        if len(self.srctypes) == 1 and len(self.dsttypes) == 1 and len(self.etypes) == 1:
            ret = 'Block(num_src_nodes={srcnode}, num_dst_nodes={dstnode}, num_edges={edge})'
            return ret.format(
                srcnode=self.number_of_src_nodes(),
                dstnode=self.number_of_dst_nodes(),
                edge=self.number_of_edges())
        else:
            ret = ('Block(num_src_nodes={srcnode},\n'
                   '      num_dst_nodes={dstnode},\n'
                   '      num_edges={edge},\n'
                   '      metagraph={meta})')
            nsrcnode_dict = {ntype : self.number_of_src_nodes(ntype)
                             for ntype in self.srctypes}
            ndstnode_dict = {ntype : self.number_of_dst_nodes(ntype)
                             for ntype in self.dsttypes}
            nedge_dict = {etype : self.number_of_edges(etype)
                          for etype in self.canonical_etypes}
            meta = str(self.metagraph().edges(keys=True))
            return ret.format(
                srcnode=nsrcnode_dict, dstnode=ndstnode_dict, edge=nedge_dict, meta=meta)
