import ast
from collections import Counter

def get_pure_src(d):
    x = ast.parse(d)
    docstring = ast.get_docstring(x.body[0], clean=False)
    return d.replace(docstring, "")

if __name__== "__main__":
    from project.data.preprocessed.overfit import overfit_data as DATA
    test = DATA.train[5]['src']
    print((test,))
    print(get_pure_src(test))

    counter = []
    for i, d in enumerate(DATA.train + DATA.valid + DATA.test):
        try:
            get_pure_src(d['src'])
        except Exception as e:
           counter.append(type(e).__name__)
           print(e.args)

           print(len(counter),i+1, d['arg_name'], type(e).__name__)

    print(Counter(counter))

