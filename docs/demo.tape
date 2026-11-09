# Records docs/demo.gif with VHS (https://github.com/charmbracelet/vhs).
# Run from the repo root, with Ollama running:  vhs docs/demo.tape
# Tip: ask Bill Buddy one question first so the model is already loaded.

Output docs/demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1280
Set Height 760
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set PlaybackSpeed 1.5

Hide
Type "source .venv/bin/activate && export PS1='$ ' && clear"
Enter
Sleep 2s
Show

Type "python -m bill_buddy"
Sleep 500ms
Enter
Wait+Screen@30s /You:/
Sleep 1s

Type "Somos 4 amigos, la cuenta fue de 230 euros y queremos dejar 10% de propina. ¿Cuánto paga cada uno?"
Sleep 500ms
Enter
Wait+Screen@120s /Bill Buddy:[\s\S]*You:/
Sleep 3s

Ctrl+L
Type "La cena costó 120 dólares, 15% de propina, dividida entre 3. ¿Cuánto es cada parte en pesos?"
Sleep 500ms
Enter
Wait+Screen@120s /Bill Buddy:[\s\S]*You:/
Sleep 3s

Type "Pesos colombianos"
Sleep 500ms
Enter
Wait+Screen@120s /Bill Buddy:[\s\S]*Bill Buddy:[\s\S]*You:/
Sleep 6s
