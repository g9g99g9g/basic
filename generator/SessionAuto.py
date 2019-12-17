TeacherList = [['YangShudong', '01100110', 3], ['ZhouYing', '11110111', 6], ['YuXueying', '01110111', 6],
               ['FuPan', '11101110', 6], ['YuFeng', '10110111', 6], ['JiaoLina', '11111111', 6],
               ['SunLin', '11111111', 1]]
StudentList = [[1, 101, 'SunA', 'SunLin'], [0, 102, 'JiaoA', 'JiaoLina'], [0, 103, 'SunB', 'SunLin'],
               [0, 104, 'YangA', 'YangShudong'], [0, 105, 'YuFA', 'YuFeng'], [1, 106, 'JiaoB', 'JiaoLina'],
               [0, 107, 'SunC', 'SunLin'], [1, 108, 'YuXA', 'YuXueying'], [0, 109, 'FuA', 'FuPan'],
               [1, 110, 'SunD', 'SunLin'], [0, 201, 'YangB', 'YangShudong'], [2, 202, 'ZhouA', 'ZhouYing'],
               [0, 203, 'ZhouB', 'ZhouYing'], [2, 204, 'YangC', 'YangShudong'], [2, 205, 'YuFB', 'YuFeng'],
               [0, 206, 'JiaoC', 'JiaoLina'], [0, 207, 'FuB', 'FuPan'], [0, 208, 'YuXB', 'YuXueying'],
               [0, 209, 'FuC', 'FuPan'], [0, 210, 'SunE', 'SunLin'], [0, 301, 'SunH', 'SunLin'],
               [0, 302, 'JiaoH', 'JiaoLina'], [0, 303, 'SunI', 'SunLin'], [1, 304, 'YangH', 'YangShudong'],
               [2, 305, 'YuFH', 'YuFeng'], [2, 306, 'JiaoH', 'JiaoLina'], [1, 307, 'SunJ', 'SunLin'],
               [0, 308, 'YuXH', 'YuXueying'], [0, 309, 'FuH', 'FuPan'], [0, 310, 'SunK', 'SunLin'],
               [0, 401, 'YangO', 'YangShudong'], [0, 402, 'ZhouO', 'ZhouYing'], [1, 403, 'ZhouP', 'ZhouYing'],
               [0, 404, 'YangP', 'YangShudong'], [0, 405, 'YuFO', 'YuFeng'], [2, 406, 'JiaoO', 'JiaoLina'],
               [1, 407, 'FuO', 'FuPan'], [0, 408, 'YuXO', 'YuXueying'], [1, 409, 'FuP', 'FuPan'],
               [0, 410, 'SunO', 'SunLin'], [0, 501, 'YangX', 'YangShudong'], [2, 502, 'ZhouX', 'ZhouYing'],
               [0, 503, 'ZhouY', 'ZhouYing'], [0, 504, 'YangY', 'YangShudong'], [1, 505, 'YuFX', 'YuFeng']]
N = 6  # 常量设置(每场Session有N个学生，M个教师，学生人数L=Count(StudentList))
M = 4

# 按StudentPriority属性对StudentList由高到低排序(同一优先属性下GROUP BY ChiefTutor)
StudentListSorted = sorted(StudentList, key=lambda StudentList:(StudentList[0], StudentList[2]))

# 场次列表SessionList(学号列表StudentNumberQueue、学生列表StudentQueue、评委列表JuryQueue)
StudentNumberQueue = list()
StudentQueue = list()
ChiefTutorTaboo = list()
JuryQueue = list()
Session = list()
Timing = list()

for i in range(0, int(len(StudentListSorted)/N)+1):
    print("\nSession", i+1, ":")
    Session = StudentListSorted[i*N:i*N+N]
    for j in range(0, len(Session)):
        StudentQueue.append(Session[j][2])
        ChiefTutorTaboo.append(Session[j][3])
    print("Students are ", StudentQueue)
    print("Tutors are ", ChiefTutorTaboo)
    for k in range(0, len(TeacherList)):
        if TeacherList[k][0] not in ChiefTutorTaboo:
            Timing = TeacherList[k][1]
            if Timing[i] == "1":
                print("Jury may be", TeacherList[k][0])
    StudentQueue.clear()
    ChiefTutorTaboo.clear()  # 清除禁忌表


'''
答辩安排自动生成器 软件设计书
【IN】
教师列表TeacherList(教师Teacher，空闲状态向量TeacherStatus[0,1]，最大可参加场次MaxAttendance)
学生列表StudentList(优先状态StudentPriority[高0,1,2低]，学号StudentNumber，学生姓名Student，第一指导教师ChiefTutor)  #为了简化列表sort()方法，需要把StudentPriority前置
常量设置(每场Session有N个学生，M个教师，学生人数L=Count(StudentList))

【OUT】
场次列表SessionList(学号列表StudentNumberQueue、学生列表StudentQueue、评委列表JuryQueue)

【约束条件】
高优先级的学生排在前面
学生的第一指导教师不能在场

【算法】
01 按StudentPriority属性对StudentList由高到低排序(同一优先属性下GROUP BY ChiefTutor)
02 for i in StudentList 
03 --StudentNumberQueue_i &= StudentNumber，StudentQueue_i &= Student  #一起抽取N个学生
04 --将对应的所有ChiefTutor计入禁忌表ChiefTutorTaboo_i
05 --for k in TeacherList
06 ----if (TeacherStatus_i == 1) && (Teacher NOT IN ChiefTutorTaboo_i)
07 ------if Attendance_k <= MaxAttendance 
08 --------JuryQueue_i &= Teacher_k
09 --------Attendance_k += 1

【验证集】
TeacherList = [['YangShudong', '01100110', 3], ['ZhouYing', '11110111', 6], ['YuXueying', '01110111',6], ['FuPan', '11101110', 6], ['YuFeng', '10110111', 6], ['JiaoLina', '11111111',6], ['SunLin', '11111111',1]]

StudentList = [[1, 101, 'SunA', 'SunLin'], [0, 102, 'JiaoA', 'JiaoLina'], [0, 103, 'SunB', 'SunLin'], [0, 104, 'YangA', 'YangShudong'], [0, 105, 'YuFA', 'YuFeng'], [1, 106, 'JiaoB', 'JiaoLina'], [0, 107, 'SunC', 'SunLin'], [1, 108, 'YuXA', 'YuXueying'], [0, 109, 'FuA', 'FuPan'], [1, 110, 'SunD', 'SunLin'], [0, 201, 'YangB', 'YangShudong'], [2, 202, 'ZhouA', 'ZhouYing'], [0, 203, 'ZhouB', 'ZhouYing'], [2, 204, 'YangC', 'YangShudong'], [2, 205, 'YuFB', 'YuFeng'], [0, 206, 'JiaoC', 'JiaoLina'], [0, 207, 'FuB', 'FuPan'], [0, 208, 'YuXB', 'YuXueying'], [0, 209, 'FuC', 'FuPan'], [0, 210, 'SunE', 'SunLin'], [0, 301, 'SunH', 'SunLin'], [0, 302, 'JiaoH', 'JiaoLina'], [0, 303, 'SunI', 'SunLin'], [1, 304, 'YangH', 'YangShudong'], [2, 305, 'YuFH', 'YuFeng'], [2, 306, 'JiaoH', 'JiaoLina'], [1, 307, 'SunJ', 'SunLin'], [0, 308, 'YuXH', 'YuXueying'], [0, 309, 'FuH', 'FuPan'], [0, 310, 'SunK', 'SunLin'], [0, 401, 'YangO', 'YangShudong'], [0, 402, 'ZhouO', 'ZhouYing'], [1, 403, 'ZhouP', 'ZhouYing'], [0, 404, 'YangP', 'YangShudong'], [0, 405, 'YuFO', 'YuFeng'], [2, 406, 'JiaoO', 'JiaoLina'], [1, 407, 'FuO', 'FuPan'], [0, 408, 'YuXO', 'YuXueying'], [1, 409, 'FuP', 'FuPan'], [0, 410, 'SunO', 'SunLin'], [0, 501, 'YangX', 'YangShudong'], [2, 502, 'ZhouX', 'ZhouYing'], [0, 503, 'ZhouY', 'ZhouYing'], [0, 504, 'YangY', 'YangShudong'], [1, 505, 'YuFX', 'YuFeng']]
'''