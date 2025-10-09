读者优先:
读者:
p(read_lock)
if ++ readcount == 1:
    p(readwritelock)
v(read_lock)
读内容
p(read_lock)
if --read_lock == 0:
    v(readwritelock)
v(read_lock)

写者:
p(read_writelock)
写
v(read_writelock)

读写公平:

读者:

p(no_writer)
    p(read_lock)
    if ++ read_count == 1:
        p(no_reader)
    v(read_lock)
v(no_writer)
    读内容
    p(read_lock)
    if -- read_count == 0:
        v(no_reader)
    v(read_lock)


写者:
p(no_reader)
 p(write_lock)
 if ++ write_count == 1:
     p(no_writer)
 v(write_lock)
v(no_reader)
 p(writer_writelock)
 写内容
 p(write_lock)
 if -- write_count == 0:
     v(no_writer)
 v(write_lock)
 v(writer_writelock)



10.一座桥很窄，一次只能通过一辆车，中间没有会车点，用 pv 实现：
（a）若对面没有车排队，则可以连续通过。
（b）若对面有车排队，则双方依次一辆一辆的通过。（具体意思就是甲方走一
辆，乙方走一辆，再甲方走一辆……）











