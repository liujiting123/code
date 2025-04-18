import numpy as np
import sympy
from sympy import symbols, sin, cos, log, exp, sqrt, cbrt

# 定义符号变量，常用于转换过程中
C, x1, x2, x3, x4, x5, x6 = symbols('C x1 x2 x3 x4 x5 x6', real=True, positive=True)

# 定义词汇表，每一行 [token名称, 采样权重, 操作元数]
MY_VOCAB = np.array([
    ['add', 4, 2],  # 二元操作符：加法
    ['mul', 6, 2],  # 二元操作符：乘法

    ['sin', 1, 1],  # 一元操作符：正弦函数
    ['cos', 1, 1],  # 一元操作符：余弦函数
    ['log', 2, 1],  # 一元操作符：对数
    ['exp', 2, 1],  # 一元操作符：指数
    ['neg', 0, 1],  # 一元操作符：取负（权重为0表示此处不做采样，可根据需要调整）
    ['inv', 3, 1],  # 一元操作符：求倒数
    ['sq', 2, 1],   # 一元操作符：平方
    ['cb', 0, 1],   # 一元操作符：立方（权重为0暂不采样）
    ['sqrt', 2, 1], # 一元操作符：平方根
    ['cbrt', 0, 1], # 一元操作符：立方根（权重为0暂不采样）
    ['C', 8, 0],    # 叶子节点：常数
    ['x1', 8, 0],   # 叶子节点：变量1
    ['x2', 8, 0],   # 叶子节点：变量2
    ['x3', 4, 0],   # 叶子节点：变量3
    ['x4', 4, 0],   # 叶子节点：变量4
    ['x5', 2, 0],   # 叶子节点：变量5
    ['x6', 2, 0],   # 叶子节点：变量6
])

def generate_expression(vocab):
    """
    递归函数：根据预定义的词汇表随机生成表达式的 token 序列。

    参数:
        vocab: 一个 numpy 数组，每一行定义一个 token 及其属性。
    返回:
        expr: token 序列（列表），按先序遍历存储表达式树。
    """
    # 按照 vocab 中的第二列权重计算采样概率
    weights = vocab[:, 1].astype('float32')
    probs = weights / np.sum(weights)
    N = len(vocab)
    expr = []
    # 根据概率从 vocab 中随机选择一个 token
    rand_idx = np.random.choice(N, p=probs)
    cur_token = vocab[rand_idx, 0]
    cur_arity = int(vocab[rand_idx, 2])
    expr.append(cur_token)

    # 如果当前 token 为叶子节点（操作数个数为 0），返回当前 token 序列
    if cur_arity == 0:
        return expr
    else:
        # 为了避免连续采样同一类一元操作符（如 sin, cos），这里做特殊处理
        if cur_token in ['sin', 'cos']:
            # 删除 sin 和 cos（使得同一表达式中只出现其中一种）
            idx1 = np.where(vocab[:, 0]=='sin')[0][0]
            idx2 = np.where(vocab[:, 0]=='cos')[0][0]
            new_vocab = np.delete(vocab, [idx1, idx2], axis=0)
        elif cur_token in ['log', 'exp']:
            idx1 = np.where(vocab[:, 0]=='log')[0][0]
            idx2 = np.where(vocab[:, 0]=='exp')[0][0]
            new_vocab = np.delete(vocab, [idx1, idx2], axis=0)
        else:
            new_vocab = vocab

        # 根据当前 token 的操作元数递归生成子表达式
        if cur_arity == 1:
            child = generate_expression(new_vocab)
            return expr + child
        elif cur_arity == 2:
            child1 = generate_expression(new_vocab)
            child2 = generate_expression(new_vocab)
            return expr + child1 + child2

def from_sequence_to_sympy(expr):
    """
    递归函数：将 token 序列转换为对应的 SymPy 表达式。

    参数:
        expr: token 序列（列表），先序遍历表示的表达式树。
    返回:
        一个 SymPy 表达式
    """
    cur_token = expr[0]

    # 如果当前 token 是数字（常数已被计算成数字），直接返回
    try:
        return float(cur_token)
    except ValueError:
        # 在 MY_VOCAB 中查找当前 token 的操作元数
        cur_idx = np.where(MY_VOCAB[:, 0] == cur_token)[0][0]
        cur_arity = int(MY_VOCAB[cur_idx, 2])

    # 对于叶子节点，返回符号对应变量或常数 C
    if cur_arity == 0:
        if cur_token == 'C':
            return C
        elif cur_token == 'x1':
            return x1
        elif cur_token == 'x2':
            return x2
        elif cur_token == 'x3':
            return x3
        elif cur_token == 'x4':
            return x4
        elif cur_token == 'x5':
            return x5
        elif cur_token == 'x6':
            return x6
    # 对一元操作符，根据递归调用构造子表达式
    elif cur_arity == 1:
        sub_expr = from_sequence_to_sympy(expr[1:])
        if cur_token == 'sin':
            return sin(sub_expr)
        elif cur_token == 'cos':
            return cos(sub_expr)
        elif cur_token == 'log':
            return log(sub_expr)
        elif cur_token == 'exp':
            return exp(sub_expr)
        elif cur_token == 'neg':
            return -sub_expr
        elif cur_token == 'inv':
            return 1 / sub_expr
        elif cur_token == 'sq':
            return sub_expr ** 2
        elif cur_token == 'cb':
            return sub_expr ** 3
        elif cur_token == 'sqrt':
            return sqrt(sub_expr)
        elif cur_token == 'cbrt':
            return cbrt(sub_expr)
    # 对二元操作符，需要根据表达式树结构将子表达式分成左右两部分
    elif cur_arity == 2:
        # 这里采用计数的方法确定左右子树的划分位置
        arity_count = 1
        idx_split = 1
        for temp_token in expr[1:]:
            try:
                float(temp_token)
                arity_count += -1
            except ValueError:
                temp_idx = np.where(MY_VOCAB[:, 0] == temp_token)[0][0]
                arity_count += int(MY_VOCAB[temp_idx, 2]) - 1
            idx_split += 1
            if arity_count == 0:
                break
        left_list = expr[1:idx_split]
        right_list = expr[idx_split:]
        if cur_token == 'add':
            return from_sequence_to_sympy(left_list) + from_sequence_to_sympy(right_list)
        elif cur_token == 'sub':
            return from_sequence_to_sympy(left_list) - from_sequence_to_sympy(right_list)
        elif cur_token == 'mul':
            return from_sequence_to_sympy(left_list) * from_sequence_to_sympy(right_list)

def expression_tree_depth(sympy_expr):
    """
    递归函数：计算给定 SymPy 表达式的树深度。

    参数:
        sympy_expr: 一个 SymPy 表达式
    返回:
        树的最大深度（整数）
    """
    if len(sympy_expr.args) == 0:
        return 1
    elif len(sympy_expr.args) == 1:
        return 1 + expression_tree_depth(sympy_expr.args[0])
    else:
        depths = [expression_tree_depth(arg) for arg in sympy_expr.args]
        return 1 + max(depths)

def sample_from_sympy_expression(sympy_expr, nb_samples=50,nb_vars=6):
    """
    根据给定的 SymPy 表达式采样数据，生成一个 tabular 数据集。

    参数:
        sympy_expr: 一个 SymPy 表达式，形如 y = f(x1, ..., xK)
        nb_samples: 采样数（行数），这里默认50
    返回:
        (np_y, np_x)：
         - np_y：函数输出的数值数组（形状为 (nb_samples,)）
         - np_x：输入变量数组，形状为 (nb_samples, 6)，不足的变量填 0
           这里变量 x1 至 xK 在 [10^-1, 10^1] 之间采用 log-uniform 采样
    """
    # 对常数 C 均匀采样：范围 [-100, 100]
    C_val = np.random.uniform(-100, 100)

    # 对变量 x1~x6 采用 log-uniform 采样：先采样对数值（均匀在 [-1,1]），再取10的幂

    log_samples = np.random.uniform(-1, 1, size=(nb_samples, nb_vars))
    np_x = 10 ** log_samples  # 采样结果  这里全都是正数，后面研究阶段回来看的时候要注意修改

    # 构造符号列表用于 lambdify：注意只传入实际变量
    # 这里默认使用 x1,..., x6，后面可以根据表达式中实际存在的变量裁剪
    f = sympy.lambdify([x1, x2, x3, x4, x5, x6, C], sympy_expr)

    try:
        # 计算 y 的值：传入所有变量和常数
        np_y = f(np_x[:, 0], np_x[:, 1], np_x[:, 2], np_x[:, 3], np_x[:, 4], np_x[:, 5], C_val)
    except Exception as e:
        # 如果采样出错（例如 log 负数），返回 None
        print("采样错误：", e)
        return None, None

    # 将 np_y 和 np_x 整理成 tabular 数据：第一列为 y，其余为 x1~x6

    data = np.hstack((np_y.reshape(-1, 1), np_x))

    # 如果数据中存在极端输出（例如 |y| > 1e9），也认为采样失败
    if np.any(np.abs(np_y) > 1e9):
        return None, None

    return  data,sympy_expr

def count_nb_variables_sympy_expr(sympy_expr):
    """
    计算 SymPy 表达式中出现的变量个数，要求变量按照 x1, x2, ... 顺序编号。

    参数:
        sympy_expr: 一个 SymPy 表达式
    返回:
        变量个数（整数）
    """
    nb_variables = 0
    expr_str = str(sympy_expr)
    while f'x{nb_variables+1}' in expr_str:
        nb_variables += 1
    return nb_variables

def from_sympy_to_sequence(sympy_expr):
    """
    Recursive function!
    Convert a SymPy expression into a standardized sequence of tokens,
    which will be used as the ground truth to train the ST.
    This function calls from_sympy_power_to_sequence,
    from_sympy_multiplication_to_sequence, and
    from_sympy_addition_to sequence.
    """
    if len(sympy_expr.args)==0:  # leaf
        return [str(sympy_expr)]
    elif len(sympy_expr.args)==1:  # unary operator
        return [str(sympy_expr.func)] + from_sympy_to_sequence(sympy_expr.args[0])
    elif len(sympy_expr.args)>=2:  # binary operator
        if sympy_expr.func==sympy.core.power.Pow:
            power_seq = from_sympy_power_to_sequence(sympy_expr.args[1])
            return power_seq + from_sympy_to_sequence(sympy_expr.args[0])
        elif sympy_expr.func==sympy.core.mul.Mul:
            return from_sympy_multiplication_to_sequence(sympy_expr)
        elif sympy_expr.func==sympy.core.add.Add:
            return from_sympy_addition_to_sequence(sympy_expr)


def from_sympy_power_to_sequence(exponent):
    """
    C.f. from_sympy_to_sequence function.
    Standardize the sequence of tokens for power functions.
    """
    if exponent==(-4):
        return ['inv', 'sq', 'sq']
    elif exponent==(-3):
        return ['inv', 'cb']
    elif exponent==(-2):
        return ['inv', 'sq']
    elif exponent==(-3/2):
        return ['inv', 'cb', 'sqrt']
    elif exponent==(-1):
        return ['inv']
    elif exponent==(-1/2):
        return ['inv', 'sqrt']
    elif exponent==(-1/3):
        return ['inv', 'cbrt']
    elif exponent==(-1/4):
        return ['inv', 'sqrt', 'sqrt']
    elif exponent==(1/4):
        return ['sqrt', 'sqrt']
    elif exponent==(1/3):
        return ['cbrt']
    elif exponent==(1/2):
        return ['sqrt']
    elif exponent==(3/2):
        return ['cb', 'sqrt']
    elif exponent==(2):
        return ['sq']
    elif exponent==(3):
        return ['cb']
    elif exponent==(4):
        return ['sq', 'sq']
    else:
        return ['abort']


def from_sympy_multiplication_to_sequence(sympy_mul_expr):
    """
    C.f. from_sympy_to_sequence function.
    Standardize the sequence of tokens for multiplications.
    """
    tokens = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    nb_factors = 0
    nb_constants = 0
    is_neg = False
    for n in range(len(sympy_mul_expr.args)):
        cur_fact = sympy_mul_expr.args[n]
        if cur_fact==(-1):
            is_neg = True
        if any(t in str(cur_fact) for t in tokens):
            nb_factors += 1
        else:
            nb_constants += 1
    seq = []
    if is_neg:
        seq.append('neg')
    for _ in range(nb_factors-1):
        seq.append('mul')
    if nb_constants>0:
        seq.append('mul')
        seq.append('C')
    for n in range(len(sympy_mul_expr.args)):
        cur_fact = sympy_mul_expr.args[n]
        if any(t in str(cur_fact) for t in tokens):
            seq = seq + from_sympy_to_sequence(cur_fact)
    return seq


def from_sympy_addition_to_sequence(sympy_add_expr):
    """
    C.f. from_sympy_to_sequence function.
    Standardize the sequence of tokens for additions.
    """
    tokens = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    nb_terms = 0
    nb_constants = 0
    for n in range(len(sympy_add_expr.args)):
        cur_term = sympy_add_expr.args[n]
        if any(t in str(cur_term) for t in tokens):
            nb_terms += 1
        else:
            nb_constants += 1
    seq = []
    for _ in range(nb_terms-1):
        seq.append('add')
    if nb_constants>0:
        seq.append('add')
        seq.append('C')
    for n in range(len(sympy_add_expr.args)):
        cur_term = sympy_add_expr.args[n]
        if any(t in str(cur_term) for t in tokens):
            seq = seq + from_sympy_to_sequence(cur_term)
    return seq


# ============================
# 示例：生成表达式及采样数据
# ============================


def create_id_vocab(vocab):
    tokens = [row[0] for row in vocab]
    id_vocab = dict()
    for i,token in enumerate(tokens):
        id_vocab[token] = i
    return id_vocab

id_vocab = create_id_vocab(MY_VOCAB)

def token2id(tokens):
    ids = []
    for token in tokens:
        id = id_vocab[token]
        ids.append(id)
    return ids


def datasets_generator(N_orig,repeat_sampling,generator_vocab=MY_VOCAB):
        # 生成一定数量的表达式（例如1000个原始表达式）
        generated_exprs = []
        #N_orig = 1000
        for _ in range(N_orig):
            expr_seq = generate_expression(generator_vocab)
            # 只保留 token 数量不超过30的表达式
            if len(expr_seq) > 30:
                continue
            # 排除仅由单一叶子组成的表达式
            if len(expr_seq) == 1:
                continue
            # 排除不含常数 C 的表达式
            if 'C' not in expr_seq:
                continue
            # 排除没有出现变量（x1~x6）的表达式
            if not any(token in expr_seq for token in ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']):
                continue
            generated_exprs.append(expr_seq)

        print("原始生成表达式数：", N_orig)
        print("过滤后表达式数：", len(generated_exprs))

        # 去重：将先序 token 序列转为字符串做唯一性判断
        unique_expr_strs = set([" ".join(expr) for expr in generated_exprs])
        unique_exprs = [expr.split(" ") for expr in unique_expr_strs]
        print("去重后唯一表达式数：", len(unique_exprs))

        # 将部分表达式转换为 SymPy 表达式，并采样生成 tabular 数据集
        count_valid_dataset = 0
        all_datasets = []  # 存放所有生成的数据集
        # 例如，对于每个表达式重复采样 5 次（实际论文中是100次，每个含50个样本）
        #repeat_sampling = 5
        for expr_seq in unique_exprs:
            try:
                sympy_expr = from_sequence_to_sympy(expr_seq)
            except Exception as e:
                print("转换出错：", expr_seq, e)
                continue

            # 可以选择对表达式做化简（提高采样成功率）
            sympy_expr = sympy.simplify(sympy_expr)
            # 如果表达式的深度太浅，则跳过
            if expression_tree_depth(sympy_expr) < 2:
                continue

            # 对于每个表达式重复采样 repeat_sampling 次
            for i in range(repeat_sampling):
                data ,expr = sample_from_sympy_expression(sympy_expr, nb_samples=50)

                if data is not None and not np.isnan(data).any():   # 确保 data 没有 NaN

                    prefix_expr = from_sympy_to_sequence(expr)
                    for i in range(len(prefix_expr)-1):
                        if prefix_expr[i] == '-1':
                            prefix_expr[i] = 'neg'
                        elif prefix_expr[i] == 'pow':
                            prefix_expr[i] = 'sq'
                        else:
                            pass

                    if 'abort' in prefix_expr:
                        continue

                    expr_id = token2id(prefix_expr)
                    final_data = [data,expr_id]
                    all_datasets.append(final_data)
                    count_valid_dataset += 1
        print("采样到的数据集数：", count_valid_dataset)
        return all_datasets

        # 现在 all_datasets 中存放着生成的 tabular 数据和对应方程式，
        # 每个数值数据集 shape: (50, 7) —— 第一列 y，后面6列为 x1~x6


