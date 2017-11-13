import sys
import yaml

document = ''

while True:
  line = sys.stdin.readline()

  if line == '':
    break

  if line.find('confusion_matrix') >= 0:
    continue

  if line.find('precision') >= 0:
    continue

  if line.find('recall') >= 0:
    continue

  if line.find('|') >= 0:
    continue

  if line.find('#') >= 0:
    document += line

  if line.find(':') >= 0:
    document += line

if len(sys.argv) > 1 and sys.argv[1] == '--yaml':
  print document
  exit(0)

document = yaml.load(document)

print 'base_dados,metodo,F-medidaMacro,F-medidaMicro'

for dataset in document:
  for method in document[dataset]:
    m = document[dataset][method]
    try:
      macro = m['macro'][0]['f1']
      micro = m['micro'][0]['f1']

      print '{},{},{},{}'.format(dataset, method, macro, micro)
    except:
      pass
