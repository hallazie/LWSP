def avg_per(logname):
	with open(logname) as f:
		lines = f.readlines()
		cnt, ttl = 0, 0
		for l in lines:
			if 'Speed: ' not in l:
				continue
			per = float(l.split('Speed: ')[1].split(' samples/sec')[0])
			ttl += per
			cnt += 1
		print ttl/float(cnt)

def cnt_c(w, k, inc, ouc):
	return w*w*k*k*inc*ouc

def cnt_btnk(inc, ouc, t, w):
	l1 = w*w*inc*ouc*t
	l2 = w*w*3*3*ouc*t
	l3 = w*w*ouc*ouc
	return l1+l2+l3

def calculate_mine():
	res = cnt_c(80,3,3,32)+cnt_btnk(32,16,4,40)+cnt_btnk(16,16,4,40)+cnt_btnk(16,24,4,20)+cnt_btnk(24,24,4,20)*2+cnt_btnk(24,32,4,10)+cnt_btnk(32,32,4,10)*3+cnt_btnk(32,64,4,10)+cnt_btnk(64,64,4,10)
	print res

def calculate_mobilenetv2():
	res = cnt_c(80,3,3,32)+cnt_btnk(32,16,1,40)+cnt_btnk(16,24,6,40)+cnt_btnk(24,24,6,40)+cnt_btnk(24,32,6,20)+cnt_btnk(32,32,6,20)*2+cnt_btnk(32,64,6,10)+cnt_btnk(64,64,6,10)*3+cnt_btnk(64,96,6,5)+cnt_btnk(96,96,6,5)*2+cnt_btnk(96,160,6,5)+cnt_btnk(160,160,6,5)*2+cnt_c(5,1,1,1280)
	print res

if __name__ == '__main__':
	avg_per('log_temp')
