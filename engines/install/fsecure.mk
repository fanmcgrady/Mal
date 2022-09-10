FSECURE_VERSION = 11.10.68
FSECURE_INSTALL_DIR = /opt/f-secure
FSECURE_UPDATE = http://download.f-secure.com/latest/fsdbupdate9.run
FSECURE_URL = https://download.f-secure.com/corpro/ls/trial/fsls-11.10.68-rtm.tar.gz
FSECURE_TMP = /tmp/fsecure

install-fsecure:	## install FSecure Linux Security
	apt install wget lib32stdc++6 rpm psmisc -y
	mkdir -p /tmp/fsecure
	wget https://download.f-secure.com/corpro/ls/trial/fsls-11.10.68-rtm.tar.gz -P /tmp/fsecure
	tar zxvf /tmp/fsecure/fsls-11.10.68-rtm.tar.gz -C /tmp/fsecure
	chmod a+x /tmp/fsecure/fsls-11.10.68-rtm/fsls-11.10.68
	/tmp/fsecure/fsls-11.10.68-rtm/fsls-11.10.68 --auto standalone lang=en --command-line-only
	make update-fsecure

update-fsecure:		## update FSecure Linux Security
	wget http://download.f-secure.com/latest/fsdbupdate9.run -P /tmp/fsecure
	mv /tmp/fsecure/fsdbupdate9.run /home/fangzhiyang/engines/f-secure
	/opt/f-secure/fsav/bin/dbupdate /home/fangzhiyang/engines/f-secure/fsdbupdate9.run
	rm -rf /tmp/fsecure

uninstall-fsecure:	## uninstall FSecure Linux Security
	/opt/f-secure/fsav/bin/uninstall-fsav
	rm -rf /opt/f-secure/

# 命令
# fsav --virus-action1=report --suspected-action1=report ~/engines/samples/notepad.exe