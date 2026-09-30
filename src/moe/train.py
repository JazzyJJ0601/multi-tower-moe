#!/usr/bin/env python3
"""Char-level Shakespeare language model training loop with MoE transformer."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from moe.tiny_transformer import TinyTransformer


def load_shakespeare(max_chars: int = 10000) -> tuple[np.ndarray, dict, dict]:
    """Return a small slice of Shakespeare text as tokenised array."""
    text = (
        "ACT I\n\nSCENE I. Athens. The palace of THESEUS.\n\n"
        "Enter THESEUS, HIPPOLYTA, PHILOSTRATE, and Attendants\n\n"
        "THESEUS:\nNow, fair Hippolyta, our nuptial hour\n"
        "Draws on apace; four happy days bring in\n"
        "Another moon: but, O, methinks, how slow\n"
        "This old moon wanes! She lingers my desires,\n"
        "Like to a step-dame or a dowager\n"
        "Long withering out a young man's revenue.\n\n"
        "HIPPOLYTA:\nFour days will quickly steep themselves in night;\n"
        "Four nights will quickly dream away the time;\n"
        "And then the moon, like to a silver bow\n"
        "New-bent in heaven, shall behold the night\n"
        "Of our solemnities.\n\n"
        "THESEUS:\nGo, Philostrate,\n"
        "Stir up the Athenian youth to merriments;\n"
        "Awake the pert and nimble spirit of mirth;\n"
        "Turn melancholy forth to funerals;\n"
        "The pale companion is not for our pomp.\n\n"
        "Enter EGEUS, HERMIA, LYSANDER, and DEMETRIUS\n\n"
        "EGEUS:\nHappy be Theseus, our renowned duke!\n\n"
        "THESEUS:\nThanks, good Egeus: what's the news with thee?\n\n"
        "EGEUS:\nFull of vexation come I, with complaint\n"
        "Against my child, my daughter Hermia.\n"
        "Stand forth, Demetrius. My noble lord,\n"
        "This man hath my consent to marry her.\n"
        "Stand forth, Lysander: and my gracious duke,\n"
        "This man hath bewitch'd the bosom of my child;\n"
        "Thou, thou, Lysander, thou hast given her rhymes,\n"
        "And interchanged love-tokens with my child:\n"
        "Thou hast by moonlight at her window sung,\n"
        "With feigning voice verses of feigning love,\n"
        "And stolen the impression of her fantasy\n"
        "With bracelets of thy hair, rings, gawds, conceits,\n"
        "Knacks, trifles, nosegays, sweetmeats, messengers\n"
        "Of strong prevailment in unharden'd youth:\n"
        "With cunning hast thou filch'd my daughter's heart,\n"
        "Turn'd her obedience, which is due to me,\n"
        "To stubborn harshness: and, my gracious duke,\n"
        "Be it so she will not here before your grace\n"
        "Consent to marry with Demetrius,\n"
        "I beg the ancient privilege of Athens,\n"
        "As she is mine, I may dispose of her:\n"
        "Which shall be either to this gentleman\n"
        "Or to her death, according to our law\n"
        "Immediately provided in that case.\n\n"
        "THESEUS:\nWhat say you, Hermia? be advised fair maid:\n"
        "To you your father should be as a god;\n"
        "One that composed your beauties, yea, and one\n"
        "To whom you are but as a form in wax\n"
        "By him imprinted and within his power\n"
        "To leave the figure or disfigure it.\n"
        "Demetrius is a worthy gentleman.\n\n"
        "HERMIA:\nSo is Lysander.\n\n"
        "THESEUS:\nIn himself he is;\n"
        "But in this kind, wanting your father's voice,\n"
        "The other must be held the worthier.\n\n"
        "HERMIA:\nI would my father look'd but with my eyes.\n\n"
        "THESEUS:\nRather your eyes must with his judgement look.\n\n"
        "HERMIA:\nI do entreat your grace to pardon me.\n"
        "I know not by what power I am made bold,\n"
        "Nor how it may concern my modesty,\n"
        "In such a presence here to plead my thoughts;\n"
        "But I beseech your grace that I may know\n"
        "The worst that may befall me in this case,\n"
        "If I refuse to wed Demetrius.\n\n"
        "THESEUS:\nEither to die the death or to abjure\n"
        "For ever the society of men.\n"
        "Therefore, fair Hermia, question your desires;\n"
        "Know of your youth, examine well your blood,\n"
        "Whether, if you yield not to your father's choice,\n"
        "You can endure the livery of a nun,\n"
        "For aye to be in shady cloister mew'd,\n"
        "To live a barren sister all your life,\n"
        "Chanting faint hymns to the cold fruitless moon.\n"
        "Thrice-blessed they that master so their blood,\n"
        "To undergo such maiden pilgrimage;\n"
        "But earthlier happy is the rose distill'd,\n"
        "Than that which withering on the virgin thorn\n"
        "Grows, lives and dies in single blessedness.\n\n"
        "HERMIA:\nSo will I grow, so live, so die, my lord,\n"
        "Ere I will yield my virgin patent up\n"
        "Unto his lordship, whose unwished yoke\n"
        "My soul consents not to give sovereignty.\n\n"
        "THESEUS:\nTake time to pause; and, by the next new moon--\n"
        "The sealing-day betwixt my love and me,\n"
        "For everlasting bond of fellowship--\n"
        "Upon that day either prepare to die\n"
        "For disobedience to your father's will,\n"
        "Or else to wed Demetrius, as he would,\n"
        "Or on Diana's altar to protest\n"
        "For aye austerity and single life.\n\n"
        "DEMETRIUS:\nRelent, sweet Hermia: and, Lysander, yield\n"
        "Thy crazed title to my certain right.\n\n"
        "LYSANDER:\nYou have her father's love, Demetrius;\n"
        "Let me have Hermia's: do you marry him.\n\n"
        "EGEUS:\nScornful Lysander! true, he hath my love,\n"
        "And what is mine my love shall render him;\n"
        "And she is mine, and all my right of her\n"
        "I do estate unto Demetrius.\n\n"
        "LYSANDER:\nI am, my lord, as well derived as he,\n"
        "As well possess'd; my love is more than his;\n"
        "My fortunes every way as fairly rank'd,\n"
        "If not with vantage, as Demetrius';\n"
        "And, which is more than all these boasts can be,\n"
        "I am beloved of beauteous Hermia:\n"
        "Why should not I then prosecute my right?\n"
        "Demetrius, I'll avouch it to his head,\n"
        "Made love to Nedar's daughter, Helena,\n"
        "And won her soul; and she, sweet lady, dotes,\n"
        "Devoutly dotes, dotes in idolatry,\n"
        "Upon this spotted and inconstant man.\n\n"
        "THESEUS:\nI must confess that I have heard so much,\n"
        "And with Demetrius thought to have spoke thereof;\n"
        "But, being over-full of self-affairs,\n"
        "My mind did lose it. But, Demetrius, come;\n"
        "And come, Egeus; you shall go with me,\n"
        "I have some private schooling for you both.\n"
        "For you, fair Hermia, look you arm yourself\n"
        "To fit your fancies to your father's will;\n"
        "Or else the law of Athens yields you up--\n"
        "Which by no means we may extenuate--\n"
        "To death, or to a vow of single life.\n"
        "Come, my Hippolyta: what cheer, my love?\n"
        "Demetrius and Egeus, go along:\n"
        "I must employ you in some business\n"
        "For our state and the progress of this time.\n"
        "Exeunt\n\n"
    )[:max_chars]

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    return data, stoi, itos


def get_batch(
    data: np.ndarray, seq_len: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample random contiguous sequences from data."""
    starts = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[s : s + seq_len] for s in starts])
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x, y


def train():
    d_model = 64
    d_ff = 128
    num_experts = 4
    top_k = 2
    seq_len = 32
    batch_size = 8
    steps = 200
    lr = 1e-2

    data, stoi, itos = load_shakespeare(max_chars=5000)
    actual_vocab = len(stoi)
    print(f"Vocabulary size: {actual_vocab}")
    print(f"Data length: {len(data)} chars")

    model = TinyTransformer(
        vocab_size=actual_vocab,
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k,
        max_seq_len=seq_len,
    )

    for step in range(steps):
        x, y = get_batch(data, seq_len, batch_size)
        logits = model.forward(x)

        # Cross-entropy loss
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        y_flat = y.reshape(-1)

        logits_stable = logits_flat - logits_flat.max(axis=-1, keepdims=True)
        exp = np.exp(logits_stable)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        loss = -np.mean(np.log(probs[np.arange(len(y_flat)), y_flat] + 1e-8))

        if step % 50 == 0:
            context = x[:1, :10]
            generated = model.generate(context, max_new=20)
            sample_text = "".join(itos[int(i)] for i in generated[0])
            print(f"Step {step:4d}, loss {loss:.4f}")
            print(f"  Sample: {sample_text!r}")

    print("\nTraining complete.")
    context = x[:1, :10]
    generated = model.generate(context, max_new=40)
    sample_text = "".join(itos[int(i)] for i in generated[0])
    print(f"Final sample: {sample_text!r}")


if __name__ == "__main__":
    train()
