qus = "hay hello how are you"
out,sof = model(qustion.split(),["start","end"])
ans = ' '.join(out)
print(ans)
