What Unicode character does chr(0) return?

> \x00

How does this character’s string representation (__repr__()) differ from its printed representation?

> repr is escaped

What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:

> It's the null character, so it doesn't print anything.

What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings

> UTF-8 is the shortest representation.

Why is this function incorrect? Provide an example of an input byte
string that yields incorrect results.

> The function is not correct because some unicode characters require more than one byte
Ex: こんにちは triggers an error

Give a two-byte sequence that does not decode to any Unicode character(s).

> c2 82 is unused.
