from p_tqdm import p_umap

def process(inp):
    out1 = inp + 1
    out2 = inp + 2
    return out1, out2

if __name__ == '__main__':
    inp = list(range(1000))
    out = zip(*p_umap(process, inp))
    breakpoint()