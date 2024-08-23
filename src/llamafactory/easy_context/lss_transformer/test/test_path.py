import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(parent_dir)
print(sys.path)
sys.path.insert(0,parent_dir)
print(sys.path)
