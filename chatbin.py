qus = input(">>> ")
out,sof = model(qustion.split(),["start","end"])
ans = ' '.join(out)
print(ans)
