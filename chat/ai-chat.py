from wxpy import *
#import time
#iBot = Bot(cache_path=True)
iBot=Bot()
iBot.file_helper.send("hello,I'm AI-Mr Rabbit!")
iBot.enable_puid()
iMyself = iBot.self
iTuling = Tuling(api_key='255ac0179223411a91866ff41829e9dd')
print('start')
iHostage = 'zz'
iFriend = iBot.friends().search(iHostage)[0]
#iFriend.send('AI Mode start...')
print('PUID is ',iFriend.puid)
@iBot.register(chats =[Friend])
def reply_my_friend(msg):
    iTuling.do_reply(msg)
reply_my_friend
embed()