#!/usr/bin/env bash
# Menu per lanciare i comandi di Personal AI Knowledge.
#
# Uso interattivo:  ./run.sh
# Uso diretto:      ./run.sh <numero> [argomenti del comando]
#
#   ./run.sh 1 ./documenti/programma --category programma
#   ./run.sh 2 "Cosa propone il programma sul lavoro?" --category programma --mode standard
#   ./run.sh 3 eval/questions.json --verbose
#
# Numeri: 1 = index, 2 = query, 3 = eval

# Si lavora sempre dalla radice del repo, dove stanno anche test.db e logs_v2.db
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

PYTHON="${PYTHON:-python3}"
CLI="src/cli.py"

# Attiva l'ambiente virtuale se esiste (Linux/macOS oppure Git Bash su Windows)
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
    # shellcheck disable=SC1091
    source .venv/Scripts/activate
fi

# Converte il numero nel nome del comando
commandName() {
    case "$1" in
        1) echo "index" ;;
        2) echo "query" ;;
        3) echo "eval" ;;
        *) return 1 ;;
    esac
}

# Lancia il comando mostrando la riga esatta che viene eseguita
runCommand() {
    echo
    printf '>>> %s %s' "$PYTHON" "$CLI"
    printf ' %q' "$@"
    echo
    echo

    "$PYTHON" "$CLI" "$@"
    local status=$?

    if [ "$status" -ne 0 ]; then
        echo
        echo "Il comando è terminato con errore (codice $status)."
    fi

    return "$status"
}

# Chiede gli argomenti facoltativi e li mette nell'array EXTRA_ARGS.
# Gli argomenti si separano con gli spazi: valori che contengono spazi non sono supportati qui.
askExtraArgs() {
    EXTRA_ARGS=()
    echo
    echo "Opzioni: $1"
    read -r -p "Argomenti aggiuntivi (invio per nessuno): " -a EXTRA_ARGS
}

interactiveIndex() {
    read -e -r -p "Cartella da indicizzare: " folder

    if [ -z "$folder" ]; then
        echo "Cartella non indicata."
        return 1
    fi

    askExtraArgs "--category {note,programma}  --strategy {sentence,word}  --dimension N  --overlap N"
    runCommand index "$folder" "${EXTRA_ARGS[@]}"
}

interactiveQuery() {
    read -r -p "Domanda: " question

    if [ -z "$question" ]; then
        echo "Domanda non indicata."
        return 1
    fi

    askExtraArgs "--category {note,programma}  --mode {strict,standard,full}  --top-k N  --min-sim X  --temperature X  --show-chunks"
    runCommand query "$question" "${EXTRA_ARGS[@]}"
}

interactiveEval() {
    read -e -r -p "File delle domande [eval/questions.json]: " file
    file="${file:-eval/questions.json}"

    askExtraArgs "--top-k N  --verbose  --rerank  --pool N  --rerank-min X"
    runCommand eval "$file" "${EXTRA_ARGS[@]}"
}

showHelp() {
    read -r -p "Di quale comando (index, query, eval; invio per l'help generale): " name

    case "$name" in
        "")                  "$PYTHON" "$CLI" --help ;;
        index|query|eval)    "$PYTHON" "$CLI" "$name" --help ;;
        *)                   echo "Comando sconosciuto: $name" ;;
    esac
}

showMenu() {
    echo
    echo "=============================="
    echo "  Personal AI Knowledge"
    echo "=============================="
    echo "  1) Indicizza una cartella"
    echo "  2) Fai una domanda"
    echo "  3) Valuta le soglie (eval)"
    echo "  4) Mostra l'help di un comando"
    echo "  0) Esci"
    echo
}

# Uso diretto: il primo argomento è il numero, gli altri vanno al comando così come sono
if [ "$#" -gt 0 ]; then
    if ! name="$(commandName "$1")"; then
        echo "Numero non valido: $1 (1 = index, 2 = query, 3 = eval)"
        exit 1
    fi

    shift
    "$PYTHON" "$CLI" "$name" "$@"
    exit $?
fi

# Il menu ha bisogno di un terminale interattivo (non funziona, per esempio, con "!bash run.sh" in Colab)
if [ ! -t 0 ]; then
    echo "Nessun terminale interattivo. Usa: ./run.sh <numero> [argomenti]"
    echo "Esempio: ./run.sh 2 \"La tua domanda\" --category programma"
    exit 1
fi

while true; do
    showMenu
    read -r -p "Scelta: " choice || break

    case "$choice" in
        1) interactiveIndex ;;
        2) interactiveQuery ;;
        3) interactiveEval ;;
        4) showHelp ;;
        0|q|Q) break ;;
        *) echo "Scelta non valida." ;;
    esac

    echo
    read -r -p "Premi invio per tornare al menu..." _ || break
done

echo "Ciao."
