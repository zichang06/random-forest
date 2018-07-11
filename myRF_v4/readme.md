# readme
有两个类，myTree和myForest，都是我实现的
这个版本，降低myTree.py的buildTree的复杂度，增加排好序的特征
使用多进程，原来的代码没有设置资源共享，导致每个进程都占1G的内存，但打印树是正常的
.1.py在尝试共享内存