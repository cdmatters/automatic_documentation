import ast
from collections import Counter

import re

def get_pure_src(d):
    src = clear_leading_indent(d['src'])
    x = ast.parse(src)
    docstring = ast.get_docstring(x.body[0], clean=True)
    if docstring is not None:
        src = clear_docstring(src, docstring)
    return src

def get_ast(d):
    # src = get_pure_src(d)
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



if __name__== "__main__":
    from project.data.preprocessed.split import split_data as DATA
    D = DATA.train + DATA.valid + DATA.test


    test = D[69]['src']
    name = D[69]
    
    src = clear_leading_indent(test)
    x = ast.parse(src)
    docstring = ast.get_docstring(x.body[0], clean=True)
    d_ = clear_docstring(src, docstring)
    print(d_, docstring, src)
    get_ast(name)


    counter = []
    for i, d in enumerate(D):
        try:
            # get_pure_src(d)
            get_ast(d)
        except Exception as e:
           counter.append(type(e).__name__)
           print(e.args)

           print(len(counter),i+1, d['arg_name'], type(e).__name__)

    print(Counter(counter))

