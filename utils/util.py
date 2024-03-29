import torch
import os


### TODO : implement this loss with parameters and mask. 


def MSEWeightedLoss(input, target, weights=None):
    if weights is None:
        weights = torch.tensor([0.1,0.1,0.1,0.1,0.1,0.1,0.4, 0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4], device=input.device)

    squared_diff = (input - target)**2
    weighted_squared_diff = squared_diff * weights.view(1, -1, 1, 1)
    loss = torch.sum(weighted_squared_diff, dim=1) 
    return torch.mean(loss)


#Just because I don't want to manage large path on git
import os

def addGitignore(path_or_file_name):
    path_gitignore = ".gitignore"
    if not os.path.exists(path_gitignore):
        with open(path_gitignore, 'w') as gitignore:
            gitignore.write("# .gitignore automatically generated. \n")
    with open(path_gitignore, 'r') as gitignore:
        readed_gitignore = gitignore.readlines()
        syntaxes = ['{}','./{}' ,'{}*', 'f{}*', '{}\ ','{}\n']
        if all(syntax.format(path_or_file_name) not in readed_gitignore for syntax in syntaxes):
            os.system(f'echo "{path_or_file_name}*" >> {path_gitignore}') 
            os.system(f'echo "{path_or_file_name}" >> {path_gitignore}') 

        else:
            print(f"{path_or_file_name} already logged in .gitignore")
            pass

