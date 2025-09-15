# Análise Comparativa c-TF-IDF dos Modelos de Linguagem

Este documento apresenta uma análise comparativa dos resultados de c-TF-IDF para os modelos `gpt-4o`, `gpt-4o-mini`, `gemini-1.5-flash` e `gemini-2.0-flash`. O objetivo é destacar as diferenças na construção de personas demográficas (gênero, raça e região) com base nas palavras mais distintivas de cada modelo.

## Nível de Abstração e Estereotipia

Observa-se uma clara progressão no nível de abstração e na complexidade das representações, com os modelos mais recentes superando estereótipos superficiais.

-   **`gemini-1.5-flash`**: Apresenta a abordagem mais estereotipada. As personas são construídas com base em associações diretas e muitas vezes redutoras:
    -   **Homem**: Ligado ao trabalho físico ("trabalho", "calejar").
    -   **Regiões**: Definidas por seus biomas e climas ("cerrado", "sertão", "floresta").
    -   **Raças**: Associadas a características físicas e conceitos genéricos de "força" e "história".

-   **`gpt-4o-mini` e `gpt-4o`**: Mostram um avanço, introduzindo mais elementos culturais.
    -   **`gpt-4o-mini`**: Começa a equilibrar aspectos físicos com culturais ("cultura", "história").
    -   **`gpt-4o`**: Aprofunda essa tendência, utilizando marcadores culturais específicos e reconhecíveis para definir as regiões (ex: "forró", "baião" para o Nordeste; "chimarrão", "gaúcho" para o Sul).

-   **`gemini-2.0-flash`**: Representa um salto significativo em sofisticação. Este modelo utiliza um léxico sociologicamente informado e politizado.
    -   **Raça Branca**: Introduz o conceito de "privilégio".
    -   **Raça Preta**: Utiliza termos como "racismo", "luta" e "resistência".
    -   **Gênero Não-Binário**: Demonstra a compreensão mais avançada, usando o pronome "elu" e conceitos como "expressão de gênero".

## Comparativo por Categoria

### Gênero

-   **Homem/Mulher**: Enquanto `gemini-1.5-flash` foca em papéis tradicionais (homem trabalhador, mulher cuidadora/forte), os outros modelos, especialmente `gemini-2.0-flash`, descrevem ambos os gêneros com termos mais abstratos como "poder" e "cultura".
-   **Não-Binário**: Há uma evolução clara. `gemini-1.5-flash` e `gpt-4o-mini` focam em termos de definição ("gênero", "identidade", "binário"). `gpt-4o` adiciona o nome neutro "Alex", e `gemini-2.0-flash` vai além, incorporando o pronome "elu" e a ideia de "desafiar normas".

### Raça

-   **Identidade Preta**: `gemini-1.5-flash` e `gpt-4o-mini` usam termos como "força" e "história". `gpt-4o` adiciona "resistência", e `gemini-2.0-flash` é o único a mencionar explicitamente "racismo" e "luta".
-   **Identidade Parda**: A representação é fluida. `gemini-1.5-flash` a associa ao "sol" e ao "nordestino". `gpt-4o` e `gemini-2.0-flash` a definem como uma "mistura" de ascendências, refletindo o debate identitário brasileiro.
-   **Identidade Indígena**: Todos os modelos conectam essa identidade à "terra", "natureza" e "ancestralidade", mas `gemini-2.0-flash` e `gpt-4o` aprofundam a dimensão cultural com termos como "língua", "rituais" e "direitos".
-   **Identidade Amarela**: A representação evolui de traços físicos ("olho amendoar" em `gemini-1.5-flash`) para uma identidade cultural explícita ("asiático", "japonês" nos modelos mais avançados).

### Região

-   **Marcadores Culturais vs. Geográficos**: `gemini-1.5-flash` depende quase exclusivamente de marcadores geográficos ("cerrado", "floresta"). `gpt-4o` e `gemini-2.0-flash` são muito mais específicos culturalmente, citando comidas ("pequi", "açaí", "churrasco"), música ("sertanejo", "forró", "carimbó") e gentílicos ("gaúcho").
-   **Especificidade**: `gemini-2.0-flash` e `gpt-4o` criam as personas regionais mais distintas e ricas em detalhes. A identidade sulista, por exemplo, é consistentemente associada ao "chimarrão", "churrasco" e "gaúcho" por ambos, enquanto a nordestina é ligada a ritmos musicais e a nortista à Amazônia. A identidade sudestina tende a ser a mais genérica e urbana ("cidade", "São Paulo") em todos os modelos.

## Conclusão

A análise comparativa do c-TF-IDF revela uma hierarquia clara na capacidade dos modelos de gerar personas complexas e nuançadas.

1.  **`gemini-1.5-flash`**: Opera com base em estereótipos físicos e geográficos.
2.  **`gpt-4o-mini` e `gpt-4o`**: Introduzem uma camada de identidade cultural, com o `gpt-4o` sendo mais específico e detalhado.
3.  **`gemini-2.0-flash`**: Demonstra a maior maturidade, empregando um vocabulário de crítica social e identidade política que reflete uma compreensão mais profunda das complexidades demográficas do Brasil.

Essa progressão sugere que os modelos mais recentes estão sendo treinados não apenas para reconhecer, mas para contextualizar social e politicamente as identidades que representam.
