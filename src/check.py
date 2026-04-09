import os

working_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(working_dir)

print(parent_dir)