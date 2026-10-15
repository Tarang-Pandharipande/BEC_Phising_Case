import pandas as pd, numpy as np, re, json, random, time
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import MaxAbsScaler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED)
d = pd.read_csv("ds/Phishing-Email-Dataset-HEAD/CEAS_08.csv")
d = d.dropna(subset=["body"]).drop_duplicates(subset=["subject", "body"])
d["subject"] = d["subject"].fillna("")
d["text"] = (d["subject"] + " " + d["body"]).str.lower()

URG = ["urgent", "immediately", "asap", "within", "today", "deadline", "overdue", "avoid", "delay", "suspend", "verify", "final notice"]
PAY = ["invoice", "payment", "wire", "transfer", "remit", "bank", "account", "iban", "swift", "funds"]
CHG = ["updated", "changed", "new account", "new bank", "change of bank", "kindly note"]
def meta(sender, subject, body, urls):
    t = (subject + " " + body).lower()
    words = max(len(t.split()), 1)
    return [urls, np.log1p(len(body)), body.count("!") / words, sum(c.isupper() for c in body) / max(len(body), 1),
            sum(t.count(w) for w in URG) / words * 100, sum(t.count(w) for w in PAY) / words * 100,
            sum(t.count(w) for w in CHG) / words * 100, int(bool(re.search(r"\d{1,3}(,\d{3})+|\$\s?\d", t)))]
META_NAMES = ["has_url", "log_len", "excl_ratio", "upper_ratio", "urgency", "payment_terms", "change_terms", "money_amount"]
M = np.array([meta(s, su, b, u) for s, su, b, u in zip(d.sender.astype(str), d.subject, d.body, d.urls)])
y = d.label.values

idx_tr, idx_te = train_test_split(np.arange(len(d)), test_size=0.2, stratify=y, random_state=SEED)
tf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, stop_words="english", sublinear_tf=True)
Xtr_t = tf.fit_transform(d.text.values[idx_tr]); Xte_t = tf.transform(d.text.values[idx_te])
sc = MaxAbsScaler(); Mtr = sc.fit_transform(M[idx_tr]); Mte = sc.transform(M[idx_te])
Xtr_h = hstack([Xtr_t, csr_matrix(Mtr)]).tocsr(); Xte_h = hstack([Xte_t, csr_matrix(Mte)]).tocsr()
ytr, yte = y[idx_tr], y[idx_te]

models = {
    "Multinomial Naive Bayes (TF-IDF)": (MultinomialNB(alpha=0.1), "t"),
    "Logistic Regression (TF-IDF)": (LogisticRegression(max_iter=2000, C=10), "t"),
    "Linear SVM (TF-IDF)": (LinearSVC(C=1.0), "t"),
    "Random Forest (meta features only)": (RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1), "m"),
    "Logistic Regression (hybrid TF-IDF + meta)": (LogisticRegression(max_iter=2000, C=10), "h"),
}
res = {}; fitted = {}
for name, (m, kind) in models.items():
    Xa, Xb = {"t": (Xtr_t, Xte_t), "m": (Mtr, Mte), "h": (Xtr_h, Xte_h)}[kind]
    t0 = time.time(); m.fit(Xa, ytr); tt = time.time() - t0
    p = m.predict(Xb)
    cm = confusion_matrix(yte, p)
    res[name] = dict(acc=accuracy_score(yte, p), prec=precision_score(yte, p), rec=recall_score(yte, p), f1=f1_score(yte, p),
                     fpr=cm[0, 1] / cm[0].sum(), train_s=tt, cm=cm.tolist())
    fitted[name] = (m, kind)
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name].items()})

# 5-fold CV F1 for best text model on the training portion
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
cvs = cross_val_score(LogisticRegression(max_iter=2000, C=10), Xtr_h, ytr, cv=cv, scoring="f1")
print("CV F1 hybrid LR", cvs.mean(), cvs.std())

# ---------- Synthetic BEC stress test ----------
vendors = [("Quanta Computer", "quantacomputer.com", "Taiwan"), ("Northwind Components", "northwindcomp.com", "USA"),
           ("Helios Optics", "heliosoptics.co.jp", "Japan"), ("Brightline Logistics", "brightline-log.com", "Germany"),
           ("Kestrel Semiconductors", "kestrelsemi.com", "South Korea"), ("Arcadia Cabling", "arcadiacabling.in", "India")]
foreign = ["Latvia", "Cyprus", "Lithuania", "Hungary", "Slovakia", "Hong Kong"]
items = ["server components", "rack hardware", "network switches", "optical modules", "storage arrays", "logistics services"]
def lookalike(dom):
    name, tld = dom.split(".", 1)
    ops = [lambda s: s.replace("m", "rn", 1) if "m" in s else s + "s", lambda s: s.replace("o", "0", 1) if "o" in s else s + "-inc",
           lambda s: s + "-billing", lambda s: s.replace("i", "l", 1) if "i" in s else s + "s", lambda s: s[:-1] if len(s) > 6 else s + "x"]
    return random.choice(ops)(name) + "." + tld
legit, bec = [], []
for i in range(60):
    v, dom, c = random.choice(vendors); it = random.choice(items); n = random.randint(1000, 9999); amt = random.randint(80, 900) * 1000
    legit.append(dict(sender=f"Accounts Receivable <ar@{dom}>", vendor=dom, country=c, bankc=c,
        subject=f"Invoice {n} for {it}",
        body=f"Dear Accounts Payable team,\nPlease find attached invoice {n} for the {it} delivered last month, per purchase order PO-{n+311}. The amount due is ${amt:,}. Payment terms are net 30 to our usual account on file.\nThank you for your continued business.\nRegards,\nFinance Department, {v}", urls=0, label=1 if False else 0))
    v, dom, c = random.choice(vendors); it = random.choice(items); n = random.randint(1000, 9999); amt = random.randint(80, 900) * 1000; fc = random.choice(foreign)
    bec.append(dict(sender=f"{v} Finance <billing@{lookalike(dom)}>", vendor=dom, country=c, bankc=fc,
        subject=random.choice([f"Invoice {n} – payment due", f"Updated remittance details – invoice {n}", f"RE: Invoice {n} for {it}"]),
        body=f"Dear Accounts Payable team,\nPlease find attached invoice {n} for the {it} shipment as per our agreement. The amount due is ${amt:,}.\n" +
             random.choice([f"Kindly note our banking details have been updated. Please remit to our new account in {fc}.",
                            f"Please be informed that due to an audit our bank account has changed. Kindly transfer the funds to our account in {fc}.",
                            f"Our previous account is under review, so kindly remit this payment to our new bank in {fc}."]) +
             "\n" + random.choice(["Payment is required within 3 business days to avoid delay of the next shipment.", "Kindly treat this as urgent and confirm once the transfer is done.", "Please process today to avoid suspension of supply."]) +
             f"\nRegards,\nFinance Manager, {v}", urls=0, label=1))
syn = pd.DataFrame(legit + bec)
syn["text"] = (syn.subject + " " + syn.body).str.lower()
Ms = sc.transform(np.array([meta(s, su, b, u) for s, su, b, u in zip(syn.sender, syn.subject, syn.body, syn.urls)]))
St = tf.transform(syn.text); Sh = hstack([St, csr_matrix(Ms)]).tocsr()
bec_res = {}
for name, (m, kind) in fitted.items():
    X = {"t": St, "m": Ms, "h": Sh}[kind]
    p = m.predict(X)
    bec_res[name] = dict(bec_detect=float(p[syn.label == 1].mean()), legit_flag=float(p[syn.label == 0].mean()))
    print("BEC", name, bec_res[name])

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
known = [v[1] for v in vendors]
def bec_feats(r):
    dom = r.sender.split("@")[1].rstrip(">")
    dist = min(lev(dom, k) for k in known)
    t = r.text
    return [int(dom not in known and dist <= 8), int(any(w in t for w in ["updated", "changed", "new account", "new bank"])),
            int(r.bankc != r.country), int(any(w in t for w in ["urgent", "within 3", "today", "avoid delay", "avoid suspension"]))]
B = np.array([bec_feats(r) for r in syn.itertuples()])
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
pr = np.zeros(len(syn))
for a, b in skf.split(B, syn.label):
    clf = LogisticRegression().fit(B[a], syn.label.values[a]); pr[b] = clf.predict(B[b])
bec_res["Hybrid BEC rules + LR (header + entity + behaviour)"] = dict(bec_detect=float(pr[syn.label == 1].mean()), legit_flag=float(pr[syn.label == 0].mean()))
print("BEC hybrid", bec_res["Hybrid BEC rules + LR (header + entity + behaviour)"])
print(syn.sender.iloc[60:66].tolist())

# ---------- Figures (blue/black only) ----------
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"], "font.size": 11})
names = list(res.keys()); short = ["NB", "LR", "SVM", "RF (meta)", "LR (hybrid)"]
fig, ax = plt.subplots(figsize=(7, 3.6))
x = np.arange(len(names)); w = 0.26
for k, (key, col) in enumerate([("prec", "#9DB9D8"), ("rec", "#3C6EA8"), ("f1", "#0B2A55")]):
    vals = [res[n][key] for n in names]
    ax.bar(x + (k - 1) * w, vals, w, color=col, label={"prec": "Precision", "rec": "Recall", "f1": "F1-score"}[key])
ax.set_xticks(x); ax.set_xticklabels(short); ax.set_ylim(0.8, 1.0); ax.set_ylabel("Score"); ax.legend(loc="lower right", frameon=False, ncol=3)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig("fig_models.png", dpi=200)
best = max(res, key=lambda n: res[n]["f1"])
cm = np.array(res[best]["cm"])
fig, ax = plt.subplots(figsize=(3.8, 3.3))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2): ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["Legitimate", "Phishing"]); ax.set_yticklabels(["Legitimate", "Phishing"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); fig.tight_layout(); fig.savefig("fig_cm.png", dpi=200)
fig, ax = plt.subplots(figsize=(7, 3.4))
bn = list(bec_res.keys()); bs = short + ["Hybrid BEC"]
ax.bar(np.arange(len(bn)) - 0.18, [bec_res[n]["bec_detect"] for n in bn], 0.36, color="#0B2A55", label="BEC emails detected")
ax.bar(np.arange(len(bn)) + 0.18, [bec_res[n]["legit_flag"] for n in bn], 0.36, color="#9DB9D8", label="Legitimate invoices wrongly flagged")
ax.set_xticks(np.arange(len(bn))); ax.set_xticklabels(bs); ax.set_ylim(0, 1.05); ax.set_ylabel("Rate"); ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig("fig_bec.png", dpi=200)

json.dump(dict(n=len(d), n_phish=int(y.sum()), n_legit=int((y == 0).sum()), n_train=len(idx_tr), n_test=len(idx_te), res=res, best=best,
               cv=[float(cvs.mean()), float(cvs.std())], bec=bec_res, vocab=len(tf.vocabulary_)), open("results.json", "w"), indent=1)
