# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 17:26:41 2018

@author: guan
"""

# coding:utf8
"""
日报
"""
import smtplib
import os
import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text  import MIMEText


class MyEmail:
    def __init__(self):
        self.user = None
        self.passwd = None
        self.to_list = []
        self.cc_list = []
        self.tag = None
        self.doc = None
        self.content=None

    def send(self):
        """
        发送邮件
        """
        try:
            server = smtplib.SMTP_SSL("smtp.exmail.qq.com", port=465)
            server.set_debuglevel(1)
#            server.starttls()
            server.ehlo()
            server.login(self.user, self.passwd)
            server.sendmail(self.user, self.to_list+self.cc_list,self.get_attach())
            server.close()
            print("send email successful")
        except Exception as e:
            print("send email failed %s" % e)

    def get_attach(self):
        """
        构造邮件内容
        """
        attach = MIMEMultipart()
        if self.tag is not None:
            # 主题,最上面的一行
            attach["Subject"] = self.tag
        if self.user is not None:
            # 显示在发件人
            attach["From"] = "%s" % self.user
        if self.to_list:
            # 收件人列表
            attach["To"] = ";".join(self.to_list)
        if self.cc_list:
            # 抄送列表
            attach["Cc"] = ";".join(self.cc_list)
        if self.doc:
            # 估计任何文件都可以用base64，比如rar等
            # 文件名汉字用gbk编码代替
            name = os.path.basename(self.doc)
            f = open(self.doc, "rb")
            xlsxApart = MIMEApplication(f.read())
            xlsxApart.add_header('Content-Disposition', 'attachment', filename=("gbk","",name))
#            xlsxApart["Content-Type"] = 'application/octet-stream'
#            xlsxApart["Content-Disposition"] = 'attachment; filename="' + name + '"'
            attach.attach(xlsxApart)
            mailMessageText=self.content
            attach.attach(MIMEText(mailMessageText, 'plain', 'utf-8'))
            f.close()
        return attach.as_string()

def getyestoday():
    today=datetime.date.today()  
    oneday=datetime.timedelta(days=1)
    yesterday=today-oneday 
    strdate=yesterday.strftime('%Y%m%d')
    m=yesterday.month
    d=yesterday.day
    return strdate,m,d

if __name__ == "__main__":
    strdate,m,d=getyestoday()
    aimname=r'F:\share\反欺诈数据\历史数据\反欺诈数据%s-%s.xlsx'%(m,d)
    try:
        dirname = r'F:\share\反欺诈数据\反欺诈数据.xlsx'
        os.rename(dirname,aimname)
    except:
        pass
    content_text='''
     Dear All:\n附件为%s-%s的欺诈日报。请查收，谢谢！\n\n\n\n-----------------------\ndata.report@tsjinrong.cn
        '''%(m,d)
    my = MyEmail()
    my.user = "data.report@tsjinrong.cn"
    my.passwd = "Data@123"    
    my.to_list =["wenjie@tsjinrong.cn","zhujunfeng@tsjinrong.cn","xiaoyongqing@tsjinrong.cn"]
    my.cc_list =["liuguangyue@tsjinrong.cn","lisimin@tsjinrong.cn","linxinyi@tsjinrong.cn","huangdengfeng@tsjinrong.cn"]
    my.tag = "反欺诈数据%s-%s"%(m,d)
    my.content=content_text
    my.doc =aimname
    my.send()