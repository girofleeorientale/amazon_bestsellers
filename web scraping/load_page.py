import urllib.request

webUrl=urllib.request.urlopen('https://www.amazon.co.uk/dp/B08LD4VXGL')

print("result: "+str(webUrl.getcode()))

htmldata=webUrl.read()

#print(htmldata)

with open('readme.txt', 'wb') as f:
    f.write(htmldata)