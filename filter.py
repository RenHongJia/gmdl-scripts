import sys

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

  if line.find('#') >= 0:
    document += line

  if line.find(':') >= 0:
    document += line

print document
