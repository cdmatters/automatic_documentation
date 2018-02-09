#!/usr/bin/python2.7
from __future__ import print_function
import ast
import symtable
from collections import namedtuple
from datetime import datetime
from tqdm import tqdm 


now = datetime.now().strftime("%d%h%y_%H%M")

def _clean_whitespace(line, do_newlines=True):
    newline = '\n' if do_newlines else ''
    return line.replace(' DCNL  ', ' DCNL ').replace('DCNL ', newline).replace('DCSP ', '    ')

def count_globals(codeblock, desc):
    FuncBody = namedtuple("FuncBody", ["codeblock", "clean", "globals", "locals", "params", "desc"])
    try:
        clean_cb =  _clean_whitespace(codeblock)
        clean_desc = _clean_whitespace(desc, do_newlines=False)
    
        sym = symtable.symtable(clean_cb, "?", "exec")
        c = sym.get_children()[0] 
        return FuncBody(codeblock, clean_cb, len(c.get_globals()), len(c.get_locals()), len(c.get_parameters()), clean_desc)
    
    except SyntaxError:
        return FuncBody(codeblock, None, -1, -1, -1, clean_desc)

def parse_db_for_globals():
    with open('data_ps.declbodies.train', 'r') as f:
        with open('data_ps.descriptions.train', 'r') as g:
            parsed, unparsed = [], []

            for l, d in tqdm(zip(f.readlines(), g.readlines())):
                cg = count_globals(l, d)
                if cg.globals >= 0:
                    parsed.append(cg)
                else:
                    unparsed.append(cg)
            
    print("Parsed: {}, Unparsed {}:".format(len(parsed),len(unparsed)))
    return parsed, unparsed

def write_parsed_to_file(parsed, global_max, param_max, locals_max, filename="parsed_functions_{}.py".format(now)):
    with open(filename, 'w') as f:
        for p in sorted(parsed, key=lambda x: (x.globals, x.params, x.locals)):
            f.write("#  G: {}, P: {}, L: {}\n".format(p.globals, p.params, p.locals))
            f.write('#  D: {}'.format(p.desc))
            f.write(p.clean)


if __name__=="__main__":
    parsed, _ = parse_db_for_globals()
    write_parsed_to_file(parsed, 4, 4, 20)

# print(table_b.get_children()[0].get_symbols()[2].is_global())

print 