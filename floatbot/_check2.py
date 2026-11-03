import ast, pathlib
for f in ['data/sst_loader.py','data/argo_loader.py','nlu/rule_parser.py','core/query_engine.py']:
    try:
        ast.parse(pathlib.Path(f).read_text(encoding='utf-8'))
        print('OK ', f)
    except SyntaxError as e:
        print('ERR', f, e)
