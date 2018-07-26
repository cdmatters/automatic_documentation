import ast
from multiprocessing import Process, JoinableQueue, cpu_count
from collections import defaultdict, namedtuple, Counter
import queue
import copy
from time import sleep
# import showast


CodePath = namedtuple("CodePath", ["from_var", "path", "to_var"])
CodePath.__repr__ = lambda s: str(s[0]) + " | " + " ".join(["{}".format(p[0]) for p in s[1]]) + " | " + str(s[2])

MAX_CODEPATH_LEN = 20

def populate_codepath_point_worker(data_queue, error_queue, new_data_queue):
    while True:
        try:
            i,d = data_queue.get()
            tree = get_ast(d)
            paths_from_root = extract_paths_from_root(tree, [], defaultdict(list))
            codepaths = extract_paths_to_leaves(d['arg_name'], paths_from_root)
            # d["codepaths"] = codepaths
            
            path_strings = []
            target_var_names = []
            for cp in codepaths:
                ps = " ".join([p[0] for p in cp.path])
                path_strings.append(ps)

                tv = cp.to_var
                target_var_names.append(tv)
                
            d["path_strings"] = path_strings
            d["target_var_string"] = target_var_names
            if codepaths:
                new_data_queue.put(d)
            else:
                error_queue.put(i)
                # print("NO PATHS in {}: name: {} pkg: {}".format(i, d['arg_name'], d['pkg']))
            data_queue.task_done()
        except SyntaxError:
            error_queue.put(i)
            # print("SYNTAX ERROR in {}: name: {} pkg: {}".format(i, d['arg_name'], d['pkg']))
            data_queue.task_done()
        except queue.Empty:
            return
    return

def pop_queue_closure(our_list):
    return_list = our_list
    def pop_queue(q):
        while True:
            try:
                item = q.get_nowait()
            
                return_list.append(item)
                q.task_done()
            except queue.Empty:
                break
    return pop_queue

def return_populated_codepath(data):
    error_queue = JoinableQueue()
    error_list = []
    new_data_queue = JoinableQueue()
    new_data_list = []

    data_queue = JoinableQueue()
    [data_queue.put_nowait((i,d)) for i, d in enumerate(data)]

    
    num_worker_threads = min(20, cpu_count() - 2)
    jobs = []
    
    print("Starting {} Processes".format(num_worker_threads))
    for _ in range(num_worker_threads - 1):
        p = Process(target=populate_codepath_point_worker, 
                                    args=(data_queue, error_queue, new_data_queue))
        p.start()
        jobs.append(p)
    
    func_lists = []
    for q, l in [(error_queue, error_list), (new_data_queue, new_data_list)]:
        func = pop_queue_closure(l)
        func_lists.append((func, q))
    
    while len(error_list) + len(new_data_list) != len(data):
        print(len(error_list), len(new_data_list), len(data), end="\r")
        [func(q) for func, q in func_lists]
    
    print(len(error_list), len(new_data_list), len(data), end="\r")
    [j.terminate() for j in jobs]
    print("Closed Processes")

    return new_data_list 

def get_pure_src(d):
    src = clear_leading_indent(d['src'])
    x = ast.parse(src)
    docstring = ast.get_docstring(x.body[0], clean=True)
    if docstring is not None:
        src = clear_docstring(src, docstring)
    return src

def get_ast(d):
    x = ast.parse(clear_leading_indent(d['src']))
    return x

def clear_docstring(src, docstring):
    d_lines = docstring.split("\n")
    s_lines = src.split("\n")

    stripped = []
    line_counter = 0 
    started_skipping = False
    seen_trip_quotes =False
    for i, s in enumerate(s_lines):
        if "'''" in s or '"""' in s:
            seen_trip_quotes=True
        if d_lines[0].strip() in s and not started_skipping and seen_trip_quotes:
            started_skipping = True
            line_counter = 0 
        
        if started_skipping:
            if line_counter >= len(d_lines):
                stripped.append(s)
            else:
                sub = d_lines[line_counter].strip()
                if sub not in s:
                    stripped.append(s)
                else:
                    subbed = s.replace(sub, "")
                    if not set(subbed).issubset({" ", "\t"}):
                        stripped.append(subbed)
                line_counter += 1
        else:
            stripped.append(s)
    return "\n".join(stripped)

def clear_leading_indent(d):
    split = d.split("\n")
    for i, char in enumerate(split[0]):
        if char != " ":
            break

    trimmed = [s[i:] for s in split]
    return "\n".join(trimmed)
    for i in range(len(split)):
        if split[i].strip().startswith("@") or split[i].strip().startswith("def"):
            split[i] = split[i].strip()
        else:
            break
    
    return "\n".join(split)


def _strip_docstring(body):
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Str):
        return body[1:]
    return body

def node_gen(node):
    attribute_gen = (getattr(node, attr) for attr in node._fields)
    return zip(node._fields, attribute_gen)

def node_type(field):
    return field.__class__.__name__, id(field)

def ignore_connecting_path(connecting_path):
    return len(connecting_path) > MAX_CODEPATH_LEN

def ignore_path(cpath, path):
    filtered_out = [
        tuple(["ClassDef","FunctionDef", "arguments", "arg"]), 
        tuple(["FunctionDef", "arguments", "arg"]), 
        tuple(["FunctionDef"])
    ]
    
    banned_ending = ['keywords', 'keyword']
    path_tuple = tuple(p[0] for p in path)
    cpath_tuple = tuple(p[0] for p in cpath)
    return path_tuple in filtered_out \
            or cpath_tuple in filtered_out \
            or cpath_tuple[-1] in banned_ending

def extract_paths_to_leaves(variable, pmap):
    core_paths = pmap[variable]

    path_tuple_list = [] # [(varX, [path,(up), fromX, (up), toY,], varY)] 
    for other_var, other_var_paths in pmap.items():
        if other_var == variable:
            continue
        for path in other_var_paths:
            for cpath in core_paths:
                if ignore_path(cpath, path): 
                    continue
                connecting_path = extract_connecting_path(cpath, path)
                if ignore_connecting_path(connecting_path):
                    continue
                path_tuple_list.append(CodePath(variable, connecting_path, other_var))
    return path_tuple_list
            
def get_root_index(pathA, pathB):
    for i in range(len(pathA)):
        if i == len(pathB):
            return 0
            break
        elif pathA[i] != pathB[i]:
            return i - 1
    return i  

def extract_connecting_path(pathA, pathB):
    root = get_root_index(pathA, pathB)

    UP = ("<-", None)
    DOWN = ("->", None)
    final = []
    for a in reversed(pathA[root:]):
        final.append(a)
        final.append(UP)
    final.pop()
    for b in pathB[root+1:]:
        final.append(DOWN)
        final.append(b)
    return final
    


def extract_paths_from_root(node, path, pmap, accept_nonvar_terminals=False):
    '''Take an AST. Return a map from variable name to list of paths to root node (ie list of lists)'''
    nodes_with_docstring = isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module))
    for name, field in node_gen(node):
        if isinstance(field, list):
            if nodes_with_docstring and name == 'body':
                field = _strip_docstring(field)
            for f in field:
                new_path = path + [(node_type(f))]
                if isinstance(f, ast.AST):
                    extract_paths_from_root(f, new_path, pmap, accept_nonvar_terminals)
                else:
                    pmap[f].append(path)
        elif isinstance(field, ast.AST):
            nt = node_type(field)
            if field._fields:
                new_path = path + [nt]
                extract_paths_from_root(field, new_path, pmap, accept_nonvar_terminals)
            else:
                # HERE WE ACCEPT NON VARIABLE TERMINALS
                if accept_nonvar_terminals:
                    pmap[nt[0]].append(path)
        elif isinstance(field, str):
            pmap[field].append(path)
        elif field is not None:
            pmap[field].append(path)
    return pmap 


if __name__== "__main__":
    from project.data.preprocessed.split import split_data as DATA
    
    # return_populated_codepath(DATA.train)
    # return_populated_codepath(DATA.valid)
    x = DATA.test

    x = DATA.train + DATA.valid + DATA.test
    x = return_populated_codepath(x)
    # printx(x)
    print(x[0]['codepaths'])

    # from project.data.preprocessed.split import split_data as DATA


    # test = D[69]['src']
    # name = D[69]
    
    # src = clear_leading_indent(test)
    # x = ast.parse(src)
    # docstring = ast.get_docstring(x.body[0], clean=True)
    # d_ = clear_docstring(src, docstring)
    # print(d_, docstring, src)
    # get_ast(name)


    # counter = []
    # for i, d in enumerate(D):
    #     try:
    #         # get_pure_src(d)
    #         get_ast(d)
    #     except Exception as e:
    #        counter.append(type(e).__name__)
    #        print(e.args)

    #        print(len(counter),i+1, d['arg_name'], type(e).__name__)

    # print(Counter(counter))

