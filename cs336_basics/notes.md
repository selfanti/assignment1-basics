# BPE分词分词组件实现

1. 预分词，按照输入的文本按照空格划分，注意到windows和linux的换行符是不一样的，应该统一使用CRLF
2. 训练BBPE算法，测试中没有使用GPT的remapping
3. 编解码的实现
