import os
import subprocess
import urllib.request
import pychu
import pychu.config

from pychu import gpu


def _dot_var(v, verbose=False):
    """v: 変数"""
    # 変数用(Tensor)
    # 変数の色はorangeの丸
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    # verbose=Trueだと変数のdtypeも出力するようにする
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)
    # id()はメモリのアドレスを返す
    return dot_var.format(id(v), name)


def _dot_func(f):
    # 関数用
    # 関数の色はlightblueの四角
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    # ここで矢印をつける
    # inputsをoutputsについてつけていく
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt


def get_dot_graph(output, verbose=True):
    # 全部をつなげる
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            # funcs.sort(key=lambda x: x.generation)
            seen_set.add(f)

    add_func(output.creator)
    # 一番最後のoutputに対して_dot_varでtxtに追加
    txt += _dot_var(output, verbose)

    # 一番最後のfuncsからinputsを使ってひとつ前の変数を取得&txtに追加
    # これを最後の変数まで繰り返す
    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g{\n' + txt + '}'


def plot_dot_graph(output, verbose=True,
                   to_file=os.path.join(pychu.config.IMG_PATH, "graph.png")):
    dot_graph = get_dot_graph(output, verbose)

    # dotデータをファイルに保存
    tmp_dir = os.path.join(os.path.expanduser("~"), ".pychu")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with open(graph_path, "w") as f:
        f.write(dot_graph)

    # dotコマンドを呼ぶ
    extension = os.path.splitext(to_file)[1][1:]
    cmd = "dot {} -T {} -o {}".format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)


def sum_to(x, shape):
    """指定したshapeに合わせてxをsumする
    Args:
        x: 入力のndarray
        shape: 出力のshape
    Returns:
        xをshapeに合わせてsumしたndarray
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def logsumexp(x, axis=1):
    """log( sum( exp(x)))を計算して返す

    Args:
        x (Tensor): 独立変数x
        axis (int): 和をとりたい方向, 0だと列方向, 1だと行方向, Defaults to 1.

    Returns:
        Tensor: log( sum( exp(x)))を計算した値
    """
    xp = gpu.get_array_module(x)
    x = xp.array(x)
    m = xp.max(x, axis=axis, keepdims=True)
    y = x - m
    y = xp.exp(y)
    s = y.sum(axis=axis, keepdims=True)
    s = xp.log(s)
    m += s
    return m


def pair(x):
    """要素が1つの時にその唯一の要素を増やして(x, x)のように返す

    Args:
        x (Tensor, k): pairにしたい(もうすでにpair)変数

    Raises:
        ValueError: xのsizeが1または2でないときにエラーをだす

    Returns:
        tuple: もとから要素数(size)が2ならそのまま, sizeが1なら(x, x)
    """
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError


def get_conv_outsize(input_size, filter_size, stride, pad):
    return (input_size + pad * 2 - filter_size) // stride + 1


def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p


# =============================================================================
# download function
# =============================================================================


def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0:
        p = 100.0
    if i >= 30:
        i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')


cache_dir = os.path.join(os.path.expanduser('~'), '.pychu')


def get_file(url, file_name=None):
    """Download a file from the `url` if it is not in the cache.

    The file at the `url` is downloaded to the `~/.dezero`.

    Args:
        url (str): URL of the file.
        file_name (str): Name of the file. It `None` is specified the original
            file name is used.

    Returns:
        str: Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt):
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path
