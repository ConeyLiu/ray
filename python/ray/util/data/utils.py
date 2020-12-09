
def divide_blocks(
    blocks: List[int],
    world_size: int) -> Dict[int, List[int]]:
    """
    Divide the blocks into world_size partitions, and return the divided block indexes for the
    given work_rank
    :param blocks: the blocks and each item is the given block size
    :param world_size: total world size
    :return: a dict, the key is the world rank, and the value the block indexes
    """
    if len(blocks) < world_size:
        raise Exception("do not have enough blocks to divide")
    results = {}
    tmp_queue = {}
    for i in range(world_size):
        results[i] = []
        tmp_queue[i] = 0
    indexes = range(len(blocks))
    blocks_with_indexes = dict(zip(indexes, blocks))
    blocks_with_indexes = dict(sorted(blocks_with_indexes.items(),
                                      key=lambda item: item[1],
                                      reverse=True))
    for i, block in blocks_with_indexes.items():
        rank = sorted(tmp_queue, key=lambda x: tmp_queue[x])[0]
        results[rank].append(i)
        tmp_queue[rank] = tmp_queue[rank] + block

    for i, indexes in results.items():
        results[i] = sorted(indexes)
    return results
