with open("train.csv", "rb") as f:
    header = f.readline()
    body = f.read()
if not body.endswith(b"\n"):
    body += b"\n"
with open("train_x5.csv", "wb") as g:
    g.write(header)
    for _ in range(5):
        g.write(body)