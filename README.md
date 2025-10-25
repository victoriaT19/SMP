-----

### 1\. Template Básico e Otimização de I/O (C++)

Este é um ponto de partida comum. A otimização de I/O (`ios_base::sync_with_stdio(0); cin.tie(0);`) é crucial para evitar Time Limit Exceeded (TLE) em problemas com muita entrada/saída.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <set>
#include <map>
#include <queue>
#include <cmath>
#include <iomanip> // Para std::setprecision

// Define macros comuns
#define ll long long
#define F first
#define S second
#define PB push_back
#define MP make_pair
#define all(x) (x).begin(), (x).end()
#define len(x) ((int)(x).size())

using namespace std;

void solve() {
    // Seu código da solução aqui
}

int main() {
    // Desabilita a sincronização com os streams de C (printf/scanf)
    // e desvincula cin de cout, acelerando I/O.
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    // Para múltiplos casos de teste
    int t = 1;
    // cin >> t; // Descomente se houver múltiplos casos
    while (t--) {
        solve();
    }

    return 0;
}
```

-----

### 2\. Teoria dos Números: Exponenciação Modular Rápida

Calcula $(base^{exp}) \pmod{mod}$ em tempo $O(\log exp)$. Essencial para problemas envolvendo aritmética modular.

```cpp
#define ll long long

/**
 * @brief Calcula (base^exp) % mod de forma eficiente.
 * * @param base A base.
 * @param exp O expoente.
 * @param mod O módulo.
 * @return (base^exp) % mod
 */
ll modPow(ll base, ll exp, ll mod) {
    ll res = 1;
    base %= mod;
    while (exp > 0) {
        // Se o bit menos significativo de exp é 1
        if (exp % 2 == 1) res = (res * base) % mod;
        
        // base = (base * base) % mod
        base = (base * base) % mod;
        
        // exp = exp / 2
        exp >>= 1;
    }
    return res;
}
```

-----

### 3\. Teoria dos Números: Crivo de Eratóstenes

Encontra todos os números primos até um limite $N$ em tempo $O(N \log \log N)$.

```cpp
#include <vector>

/**
 * @brief Encontra todos os primos até n usando o Crivo de Eratóstenes.
 * * @param n O limite superior (inclusive).
 * @return vector<bool> onde is_prime[i] é true se i é primo, false caso contrário.
 */
std::vector<bool> sieveOfEratosthenes(int n) {
    // is_prime[i] = true se i for primo.
    std::vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false; // 0 e 1 não são primos

    for (int p = 2; p * p <= n; p++) {
        // Se is_prime[p] é true, então p é primo
        if (is_prime[p]) {
            // Marca todos os múltiplos de p como não-primos
            for (int i = p * p; i <= n; i += p)
                is_prime[i] = false;
        }
    }
    return is_prime;
}

/**
 * @brief Gera uma lista de números primos até n.
 *
 * @param n O limite superior (inclusive).
 * @return vector<int> contendo os primos.
 */
std::vector<int> getPrimes(int n) {
    std::vector<bool> is_prime = sieveOfEratosthenes(n);
    std::vector<int> primes;
    for (int p = 2; p <= n; p++) {
        if (is_prime[p]) {
            primes.push_back(p);
        }
    }
    return primes;
}
```

-----

### 4\. Grafos: Algoritmo de Dijkstra

Encontra o menor caminho de um nó de origem para todos os outros nós em um grafo com pesos não-negativos. A complexidade é $O(E \log V)$ usando uma fila de prioridade.

```cpp
#include <vector>
#include <queue>
#include <map>

#define ll long long
using namespace std;

const ll INF = 1e18; // Infinito

/**
 * @brief Executa o algoritmo de Dijkstra.
 * * @param n O número de nós (de 0 a n-1).
 * @param adj Lista de adjacência (mapa de vizinho para peso).
 * @param startNode O nó inicial.
 * @return vector<ll> contendo as menores distâncias do nó inicial para todos os outros.
 */
vector<ll> dijkstra(int n, const vector<map<int, int>>& adj, int startNode) {
    // Fila de prioridade (min-heap): armazena {distância, nó}
    priority_queue<pair<ll, int>, vector<pair<ll, int>>, greater<pair<ll, int>>> pq;

    // Vetor de distâncias, inicializado com infinito
    vector<ll> dist(n, INF);

    // Distância do nó inicial para ele mesmo é 0
    dist[startNode] = 0;
    pq.push({0, startNode});

    while (!pq.empty()) {
        // Pega o nó com a menor distância
        ll d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        // Se a distância na fila for maior que a já registrada, ignora
        if (d > dist[u]) continue;

        // Itera sobre os vizinhos de u
        for (auto const& [v, weight] : adj[u]) {
            // Relaxamento da aresta (u, v)
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }

    return dist;
}

/*
// Exemplo de uso:
int main() {
    int n = 5; // 5 nós (0 a 4)
    vector<map<int, int>> adj(n);

    // adj[u][v] = peso
    adj[0][1] = 10;
    adj[0][3] = 5;
    adj[1][2] = 1;
    adj[1][3] = 2;
    adj[2][4] = 4;
    adj[3][1] = 3;
    adj[3][2] = 9;
    adj[3][4] = 2;
    adj[4][0] = 7;
    adj[4][2] = 6;

    int startNode = 0;
    vector<ll> distances = dijkstra(n, adj, startNode);

    cout << "Menores distâncias a partir do nó " << startNode << ":\n";
    for (int i = 0; i < n; ++i) {
        if (distances[i] == INF) {
            cout << "Nó " << i << ": INF\n";
        } else {
            cout << "Nó " << i << ": " << distances[i] << "\n";
        }
    }
    return 0;
}
*/
```

-----

### 5\. Estrutura de Dados: Árvore de Segmentos (Segment Tree)

Uma estrutura de dados poderosa para realizar consultas de intervalo (como soma, mínimo, máximo) e atualizações de ponto em $O(\log N)$.

Este exemplo implementa uma "SegTree" para consultas de soma de intervalo.

```cpp
#include <vector>

#define ll long long
using namespace std;

/**
 * @brief Árvore de Segmentos (Segment Tree) para consultas de soma de intervalo.
 */
struct SegTree {
    vector<ll> tree;
    int n;

    // Construtor: constrói a árvore a partir de um vetor inicial
    SegTree(const vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n); // Tamanho seguro para a árvore
        build(arr, 0, 0, n - 1);
    }

    // Função recursiva para construir a árvore
    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            // Nó folha
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            // Constrói recursivamente as subárvores esquerda e direita
            build(arr, 2 * node + 1, start, mid);
            build(arr, 2 * node + 2, mid + 1, end);
            // Nó interno armazena a soma dos filhos
            tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
        }
    }

    /**
     * @brief Atualiza o valor na posição 'idx' para 'val'.
     */
    void update(int idx, int val) {
        updateRec(0, 0, n - 1, idx, val);
    }

    // Função recursiva de atualização
    void updateRec(int node, int start, int end, int idx, int val) {
        if (start == end) {
            // Nó folha (na posição idx)
            tree[node] = val;
            return;
        }

        int mid = (start + end) / 2;
        if (start <= idx && idx <= mid) {
            // idx está na subárvore esquerda
            updateRec(2 * node + 1, start, mid, idx, val);
        } else {
            // idx está na subárvore direita
            updateRec(2 * node + 2, mid + 1, end, idx, val);
        }
        // Atualiza a soma no nó pai
        tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
    }

    /**
     * @brief Consulta a soma no intervalo [L, R] (inclusivo).
     */
    ll query(int L, int R) {
        return queryRec(0, 0, n - 1, L, R);
    }

    // Função recursiva de consulta
    ll queryRec(int node, int start, int end, int L, int R) {
        if (R < start || end < L) {
            // Intervalo [start, end] está fora de [L, R]
            return 0; // Elemento neutro da soma
        }
        if (L <= start && end <= R) {
            // Intervalo [start, end] está totalmente contido em [L, R]
            return tree[node];
        }

        // Intervalo [start, end] sobrepõe parcialmente [L, R]
        int mid = (start + end) / 2;
        ll p1 = queryRec(2 * node + 1, start, mid, L, R);
        ll p2 = queryRec(2 * node + 2, mid + 1, end, L, R);
        return p1 + p2;
    }
};

/*
// Exemplo de uso:
int main() {
    vector<int> arr = {1, 3, 5, 7, 9, 11};
    SegTree st(arr);

    // Soma do intervalo [1, 3] (índices 1, 2, 3) -> 3 + 5 + 7 = 15
    cout << "Soma de [1, 3]: " << st.query(1, 3) << endl; // Output: 15

    // Atualiza arr[2] = 6
    st.update(2, 6);

    // Soma do intervalo [1, 3] -> 3 + 6 + 7 = 16
    cout << "Soma de [1, 3] após update: " << st.query(1, 3) << endl; // Output: 16

    return 0;
}
*/
```

-----

### 1\. Estruturas de Dados: Union-Find (DSU)

Também conhecido como Disjoint Set Union (DSU), é usado para rastrear um conjunto de elementos particionados em vários subconjuntos disjuntos. É incrivelmente rápido (quase constante por operação) e essencial para o algoritmo de Kruskal, entre outros.

Esta implementação usa "Union by Size" (União por Tamanho) e "Path Compression" (Compressão de Caminho) para otimização.

```cpp
#include <vector>
#include <numeric> // Para std::iota

using namespace std;

/**
 * @brief Estrutura de Dados Union-Find (Disjoint Set Union).
 * Mantém uma coleção de conjuntos disjuntos.
 */
struct DSU {
    vector<int> parent;
    vector<int> sz; // 'sz[i]' é o tamanho da componente da qual 'i' é raiz.

    /**
     * @brief Constrói a estrutura DSU para 'n' elementos.
     * @param n Número de elementos (0 a n-1).
     */
    DSU(int n) {
        parent.resize(n);
        // Inicializa cada elemento como sua própria raiz.
        // std::iota preenche o vetor com 0, 1, 2, ..., n-1
        iota(parent.begin(), parent.end(), 0);
        
        sz.assign(n, 1); // Cada componente começa com tamanho 1.
    }

    /**
     * @brief Encontra a raiz (representante) do conjunto ao qual 'i' pertence.
     * Usa "Path Compression".
     * @param i O elemento.
     * @return A raiz do conjunto de 'i'.
     */
    int find(int i) {
        if (parent[i] == i)
            return i;
        // Path Compression: faz com que todos os nós no caminho
        // apontem diretamente para a raiz.
        return parent[i] = find(parent[i]);
    }

    /**
     * @brief Une os conjuntos que contêm os elementos 'a' e 'b'.
     * Usa "Union by Size".
     * @param a Um elemento do primeiro conjunto.
     * @param b Um elemento do segundo conjunto.
     * @return true se a união foi realizada (a e b estavam em conjuntos
     * diferentes), false caso contrário.
     */
    bool unite(int a, int b) {
        int root_a = find(a);
        int root_b = find(b);

        if (root_a != root_b) {
            // Union by Size: anexa a árvore menor à árvore maior.
            if (sz[root_a] < sz[root_b])
                swap(root_a, root_b);
            
            parent[root_b] = root_a;
            sz[root_a] += sz[root_b];
            return true;
        }
        
        return false; // 'a' e 'b' já estavam no mesmo conjunto.
    }

    /**
     * @brief Verifica se 'a' e 'b' estão no mesmo conjunto.
     */
    bool in_same_set(int a, int b) {
        return find(a) == find(b);
    }
    
    /**
     * @brief Retorna o tamanho do conjunto ao qual 'i' pertence.
     */
    int component_size(int i) {
        return sz[find(i)];
    }
};
```

-----

### 2\. Estruturas de Dados: Árvore de Fenwick (BIT)

Uma Binary Indexed Tree (BIT) ou Árvore de Fenwick é usada para calcular somas de prefixo e realizar atualizações de ponto em tempo $O(\log N)$. É mais simples e rápido de codar do que uma Segment Tree para essa tarefa específica.

**Nota:** Esta implementação usa indexação baseada em 1, o que simplifica a lógica `i & -i`.

```cpp
#include <vector>

#define ll long long
using namespace std;

/**
 * @brief Árvore de Fenwick (Binary Indexed Tree - BIT).
 * Usada para consultas de soma de prefixo e atualizações de ponto.
 * Opera com indexação baseada em 1.
 */
struct FenwickTree {
    vector<ll> bit;
    int n;

    /**
     * @brief Constrói a BIT.
     * @param size O tamanho máximo (os índices irão de 1 a size).
     */
    FenwickTree(int size) {
        n = size;
        bit.assign(n + 1, 0); // n+1 para acomodar a indexação 1-based
    }

    /**
     * @brief Adiciona 'val' ao elemento na posição 'idx'.
     * @param idx A posição (1-based).
     * @param val O valor a ser adicionado.
     */
    void update(int idx, ll val) {
        for (; idx <= n; idx += idx & -idx) {
            bit[idx] += val;
        }
    }

    /**
     * @brief Calcula a soma do prefixo [1, idx].
     * @param idx A posição final (1-based).
     * @return A soma de arr[1] + ... + arr[idx].
     */
    ll query(int idx) {
        ll sum = 0;
        for (; idx > 0; idx -= idx & -idx) {
            sum += bit[idx];
        }
        return sum;
    }

    /**
     * @brief Calcula a soma do intervalo [L, R].
     * @param L Início do intervalo (1-based).
     * @param R Fim do intervalo (1-based).
     * @return A soma de arr[L] + ... + arr[R].
     */
    ll queryRange(int L, int R) {
        return query(R) - query(L - 1);
    }
};
```

-----

### 3\. Grafos: Algoritmo de Kruskal (MST)

Encontra a Árvore Geradora Mínima (Minimum Spanning Tree - MST) de um grafo ponderado, não-direcionado e conexo. Ele usa a estrutura DSU.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric> // Para DSU

// (Inclua a struct DSU da Seção 1 aqui)
// ... struct DSU { ... };

using namespace std;
#define ll long long

/**
 * @brief Representa uma aresta em um grafo.
 */
struct Edge {
    int u, v;
    ll weight;
    
    // Operador para ordenar arestas por peso
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

/**
 * @brief Calcula o custo da Árvore Geradora Mínima (MST) usando Kruskal.
 *
 * @param n O número de nós (assumindo nós de 0 a n-1).
 * @param edges Um vetor de todas as arestas do grafo.
 * @param mst_edges (Saída) Um vetor para armazenar as arestas que compõem a MST.
 * @return O custo total da MST. Retorna -1 se o grafo não for conexo.
 */
ll kruskal(int n, vector<Edge>& edges, vector<Edge>& mst_edges) {
    // 1. Ordena todas as arestas pelo peso
    sort(edges.begin(), edges.end());

    DSU dsu(n);
    ll mst_cost = 0;
    int edges_in_mst = 0;
    mst_edges.clear();

    // 2. Itera sobre as arestas ordenadas
    for (const auto& edge : edges) {
        // 3. Se u e v não estão no mesmo conjunto, adiciona a aresta à MST
        if (dsu.unite(edge.u, edge.v)) {
            mst_cost += edge.weight;
            mst_edges.push_back(edge);
            edges_in_mst++;
        }
    }

    // 4. Verifica se a MST é válida (grafo era conexo)
    if (edges_in_mst == n - 1) {
        return mst_cost;
    } else {
        return -1; // Grafo não é conexo
    }
}
```

-----

### 4\. Algoritmos de Ordenação: Merge Sort

Embora em C++ você quase sempre use `std::sort` (que é $O(N \log N)$ e altamente otimizado), o Merge Sort é um algoritmo clássico de "Dividir para Conquistar". Tê-lo em um *codebook* é útil porque sua lógica de "merge" pode ser adaptada para resolver outros problemas, como *contar inversões* em um array.

```cpp
#include <vector>

using namespace std;

// Função auxiliar para mesclar dois subarrays ordenados
// arr[l..m] e arr[m+1..r]
void merge(vector<int>& arr, int l, int m, int r, vector<int>& temp) {
    int i = l;     // Índice inicial do primeiro subarray
    int j = m + 1; // Índice inicial do segundo subarray
    int k = l;     // Índice inicial do subarray mesclado (no temp)

    while (i <= m && j <= r) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    // Copia os elementos restantes do primeiro subarray, se houver
    while (i <= m) {
        temp[k++] = arr[i++];
    }

    // Copia os elementos restantes do segundo subarray, se houver
    while (j <= r) {
        temp[k++] = arr[j++];
    }

    // Copia o array 'temp' mesclado de volta para 'arr'
    for (k = l; k <= r; k++) {
        arr[k] = temp[k];
    }
}

// Função principal recursiva do Merge Sort
void mergeSortRecursive(vector<int>& arr, int l, int r, vector<int>& temp) {
    if (l < r) {
        int m = l + (r - l) / 2; // Evita overflow para (l+r)/2

        // Ordena a primeira e a segunda metade
        mergeSortRecursive(arr, l, m, temp);
        mergeSortRecursive(arr, m + 1, r, temp);

        // Mescla as metades ordenadas
        merge(arr, l, m, r, temp);
    }
}

/**
 * @brief Ordena um vetor usando o algoritmo Merge Sort.
 * @param arr O vetor a ser ordenado (modificado no local).
 */
void mergeSort(vector<int>& arr) {
    int n = arr.size();
    if (n > 0) {
        vector<int> temp(n); // Array temporário para mesclagem
        mergeSortRecursive(arr, 0, n - 1, temp);
    }
}
```

-----

### 5\. Programação Dinâmica: Problema da Mochila (Knapsack 0/1)

O problema clássico de PD. Dado um conjunto de itens, cada um com um peso e um valor, determine quais itens incluir em uma mochila de capacidade $W$ para que o valor total seja maximizado, sem exceder a capacidade.

```cpp
#include <vector>
#include <iostream>

using namespace std;

/**
 * @brief Resolve o problema da Mochila 0/1 (Knapsack).
 *
 * @param W Capacidade máxima da mochila.
 * @param weights Vetor com os pesos dos itens.
 * @param values Vetor com os valores dos itens.
 * @param n Número de itens.
 * @return O valor máximo que pode ser carregado na mochila.
 */
int knapsack(int W, const vector<int>& weights, const vector<int>& values, int n) {
    // dp[i][w] = valor máximo usando os primeiros 'i' itens
    // com uma capacidade máxima 'w'.
    // Usamos (n+1) e (W+1) para facilitar a indexação 1-based dos itens.
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    // Itera sobre os itens (de 1 a n)
    for (int i = 1; i <= n; ++i) {
        // O peso e o valor do item 'i' (índice i-1 nos vetores)
        int current_weight = weights[i - 1];
        int current_value = values[i - 1];

        // Itera sobre todas as capacidades possíveis (de 0 a W)
        for (int w = 0; w <= W; ++w) {
            // Caso 1: Não incluir o item 'i'.
            // O valor é o mesmo que o valor máximo com 'i-1' itens.
            dp[i][w] = dp[i - 1][w];

            // Caso 2: Incluir o item 'i', se ele couber (w >= current_weight).
            if (w >= current_weight) {
                // Compara o valor de não incluir o item (já em dp[i][w])
                // com o valor de incluí-lo.
                dp[i][w] = max(dp[i][w],
                               current_value + dp[i - 1][w - current_weight]);
            }
        }
    }

    // A resposta final está em dp[n][W]
    return dp[n][W];
}
```

-----

### 6\. Geometria Computacional: Convex Hull (Monotone Chain)

Encontra o fecho convexo (convex hull) de um conjunto de pontos. O fecho convexo é o menor polígono convexo que contém todos os pontos. O algoritmo "Monotone Chain" faz isso em $O(N \log N)$ (dominado pela ordenação inicial dos pontos).

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

#define ll long long
using namespace std;

/**
 * @brief Estrutura para representar um Ponto (ou vetor).
 */
struct Point {
    ll x, y;
    
    // Usado para ordenar pontos
    bool operator<(const Point& p) const {
        if (x != p.x) return x < p.x;
        return y < p.y;
    }
};

/**
 * @brief Calcula o produto vetorial (cross product) de 3 pontos (a, b, c).
 * É usado para determinar a orientação (curva).
 * @param a Ponto de origem.
 * @param b Ponto intermediário.
 * @param c Ponto final.
 * @return > 0 se (a, b, c) forma uma curva à esquerda (CCW - Counter-Clockwise).
 * < 0 se (a, b, c) forma uma curva à direita (CW - Clockwise).
 * = 0 se os pontos são colineares.
 */
ll cross_product(Point a, Point b, Point c) {
    // (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

/**
 * @brief Encontra o Fecho Convexo (Convex Hull) de um conjunto de pontos.
 * Usa o algoritmo Monotone Chain (Andrew's Algorithm).
 * @param points Vetor de pontos.
 * @return Um vetor de pontos que formam o fecho convexo, em ordem anti-horária.
 */
vector<Point> convex_hull(vector<Point>& points) {
    int n = points.size();
    if (n < 3) return points; // Fecho de 0, 1 ou 2 pontos são os próprios pontos

    // 1. Ordena os pontos lexicograficamente (primeiro por x, depois por y)
    sort(points.begin(), points.end());

    vector<Point> hull;

    // 2. Constrói o "lower hull" (casca inferior)
    for (int i = 0; i < n; ++i) {
        // Remove pontos enquanto a nova adição (points[i])
        // não formar uma curva à esquerda.
        while (hull.size() >= 2 && 
               cross_product(hull[hull.size() - 2], hull.back(), points[i]) <= 0) {
            // <= 0 inclui pontos colineares (use < 0 para excluir)
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }

    // Salva o tamanho do "lower hull"
    int lower_hull_size = hull.size();

    // 3. Constrói o "upper hull" (casca superior)
    // Itera de trás para frente
    for (int i = n - 2; i >= 0; --i) {
        while (hull.size() > lower_hull_size &&
               cross_product(hull[hull.size() - 2], hull.back(), points[i]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }

    // 4. Remove o último ponto (que é duplicado, igual ao primeiro)
    hull.pop_back();

    return hull;
}
```

-----

### 1\. Algoritmos de Busca: Busca Binária (no Vetor e na Resposta)

A busca binária é uma das técnicas mais importantes. Ela não se aplica apenas a vetores ordenados, mas a qualquer função "monotônica" (onde a resposta muda de "falso" para "verdadeiro" e nunca volta atrás).

#### a) Busca Binária em Vetor (Padrão STL)

Para encontrar um item `x` em um vetor `v` ordenado, o `std::lower_bound` é seu melhor amigo.

```cpp
#include <vector>
#include <algorithm>
#include <iostream>

// v deve estar ordenado!
std::vector<int> v = {2, 3, 5, 7, 11, 13, 17, 19};

// Encontra o primeiro elemento que é >= x
// Retorna um iterador para o elemento
auto it = std::lower_bound(v.begin(), v.end(), 10);

if (it != v.end()) {
    std::cout << "Primeiro elemento >= 10 é: " << *it << std::endl; // Output: 11
    int index = std::distance(v.begin(), it);
    std::cout << "No índice: " << index << std::endl; // Output: 4
} else {
    std::cout << "Nenhum elemento é >= 10" << std::endl;
}

// Para checar se 13 existe:
it = std::lower_bound(v.begin(), v.end(), 13);
if (it != v.end() && *it == 13) {
    std::cout << "13 encontrado!" << std::endl;
}
```

#### b) Template de Busca Binária na Resposta (Binsearch on Answer)

Usado quando você quer encontrar o valor mínimo (ou máximo) que satisfaz uma certa condição `check()`.

```cpp
/**
 * @brief Função de checagem (predicado) para a busca binária.
 * Deve ser monotônica (ex: F, F, F, T, T, T...)
 * @param x O valor a ser testado.
 * @return true se x satisfaz a condição, false caso contrário.
 */
bool check(int x) {
    // Exemplo: "x é grande o suficiente?"
    // Coloque sua lógica de verificação aqui.
    return (x * x >= 100); // Ex: Encontrar o menor x tal que x*x >= 100
}

/**
 * @brief Encontra o MENOR valor 'x' no intervalo [lo, hi] que satisfaz check(x).
 * Pressupõe que check(x) é monotônica (Falso -> Verdadeiro).
 * @param lo Limite inferior do espaço de busca.
 * @param hi Limite superior do espaço de busca.
 * @return O menor 'x' que satisfaz check(x).
 */
int binarySearchOnAnswer(int lo, int hi) {
    int ans = hi + 1; // Inicializa com um valor fora do intervalo (para "não encontrado")
    
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        
        if (check(mid)) {
            // Se 'mid' funciona, ele é um candidato a resposta.
            // Tentamos encontrar um valor ainda menor na metade esquerda.
            ans = mid;
            hi = mid - 1;
        } else {
            // Se 'mid' não funciona, precisamos de um valor maior.
            lo = mid + 1;
        }
    }
    return ans;
}

// Para encontrar o MAIOR valor que satisfaz check(x),
// basta inverter a lógica no 'if (check(mid))':
// ans = mid; lo = mid + 1;
// } else { hi = mid - 1; }
```

-----

### 2\. Grafos: Busca em Largura (BFS)

Usada para encontrar o menor caminho (em número de arestas) de um nó de origem para todos os outros nós em um grafo não-ponderado.

```cpp
#include <vector>
#include <queue>
#include <map>

using namespace std;
const int INF = 1e9;

/**
 * @brief Executa uma Busca em Largura (BFS) a partir de um nó inicial.
 *
 * @param n O número de nós (0 a n-1).
 * @param adj A lista de adjacência (pode ser vector<vector<int>>).
 * @param startNode O nó inicial.
 * @return Um vetor 'dist' onde dist[i] é a menor distância (em arestas)
 * de startNode até i. dist[i] == INF se 'i' for inalcançável.
 */
vector<int> bfs(int n, const vector<vector<int>>& adj, int startNode) {
    vector<int> dist(n, INF);
    queue<int> q;

    dist[startNode] = 0;
    q.push(startNode);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v : adj[u]) {
            // Se 'v' ainda não foi visitado (dist == INF)
            if (dist[v] == INF) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
    return dist;
}
```

-----

### 3\. Grafos: Busca em Profundidade (DFS) e Ordenação Topológica

DFS é o bloco de construção para muitos algoritmos de grafos: detecção de ciclos, ordenação topológica, componentes conexos, etc.

#### a) DFS Básico (Recursivo)

```cpp
#include <vector>

using namespace std;

vector<bool> visited;
vector<vector<int>> adj;

/**
 * @brief Função recursiva da Busca em Profundidade (DFS).
 * @param u O nó atual sendo visitado.
 */
void dfs(int u) {
    visited[u] = true;
    // Pré-ordem: processar 'u' antes de visitar vizinhos
    // cout << "Visitando: " << u << endl;

    for (int v : adj[u]) {
        if (!visited[v]) {
            dfs(v);
        }
    }
    // Pós-ordem: processar 'u' depois de visitar vizinhos
    // (Útil para ordenação topológica)
}

/*
// Exemplo de como chamar o DFS para percorrer todo o grafo
// (mesmo que ele seja desconexo)
int n; // número de nós
adj.resize(n);
visited.assign(n, false);
// ... preencher adj ...

for (int i = 0; i < n; ++i) {
    if (!visited[i]) {
        dfs(i); // Inicia um DFS a partir do nó i
    }
}
*/
```

#### b) Ordenação Topológica (Usando DFS)

Para Grafos Acíclicos Direcionados (DAGs). Produz uma ordenação linear dos vértices tal que para toda aresta $u \to v$, $u$ vem antes de $v$ na ordenação.

```cpp
#include <vector>
#include <algorithm>

using namespace std;

vector<bool> visited_topo;
vector<vector<int>> adj_topo;
vector<int> topo_order; // Armazena a ordem topológica

/**
 * @brief DFS modificado para salvar a ordem topológica.
 */
void dfs_topo(int u) {
    visited_topo[u] = true;
    for (int v : adj_topo[u]) {
        if (!visited_topo[v]) {
            dfs_topo(v);
        }
    }
    // Adiciona o nó à lista após todos os seus vizinhos
    // (dependentes) terem sido processados.
    topo_order.push_back(u);
}

/**
 * @brief Encontra a ordenação topológica de um DAG.
 * @param n Número de nós.
 * @return Um vetor com a ordem topológica.
 * (Note: A ordem está REVERSA, precisa inverter no final).
 */
vector<int> getTopologicalSort(int n) {
    adj_topo.resize(n);
    visited_topo.assign(n, false);
    topo_order.clear();
    // ... preencher adj_topo ...
    
    for (int i = 0; i < n; ++i) {
        if (!visited_topo[i]) {
            dfs_topo(i);
        }
    }
    
    // O resultado de 'topo_order' está em pós-ordem reversa,
    // que é a ordenação topológica.
    reverse(topo_order.begin(), topo_order.end());
    return topo_order;
}
```

-----

### 4\. Teoria dos Números: GCD, LCM e Inverso Modular

Funções pequenas, mas essenciais.

```cpp
#define ll long long

/**
 * @brief Calcula o Máximo Divisor Comum (GCD) de a e b.
 * (Em C++17, você pode usar std::gcd(a, b))
 */
ll gcd(ll a, ll b) {
    while (b) {
        a %= b;
        swap(a, b);
    }
    return a;
}

/**
 * @brief Calcula o Mínimo Múltiplo Comum (LCM) de a e b.
 * (Em C++17, você pode usar std::lcm(a, b))
 * Usa (a * b) / gcd(a, b) de forma segura contra overflow.
 */
ll lcm(ll a, ll b) {
    if (a == 0 || b == 0) return 0;
    return (a / gcd(a, b)) * b;
}

// (Depende da função modPow(base, exp, mod) da resposta anterior)
ll modPow(ll base, ll exp, ll mod); // Assumindo que você a tenha

/**
 * @brief Calcula o inverso modular de n (n^-1) módulo m.
 * Apenas funciona se 'm' for um número primo (Pelo Pequeno Teorema de Fermat).
 * @param n O número.
 * @param mod O módulo (deve ser primo).
 * @return (n^(mod-2)) % mod
 */
ll modInverse(ll n, ll mod) {
    return modPow(n, mod - 2, mod);
}
```

-----

### 5\. String: Hashing de String (Algoritmo Rabin-Karp)

Extremamente útil para comparar substrings em $O(1)$ (após um pré-processamento $O(N)$). Permite encontrar duplicatas, palíndromos, etc.

**Aviso:** Hashing tem chance (muito pequena) de colisão. Em competições, geralmente é seguro, mas às vezes é necessário usar dois *hashes* (dois `base` e dois `mod`) para garantir.

```cpp
#include <string>
#include <vector>

#define ll long long
using namespace std;

/**
 * @brief Estrutura para calcular hashes de substrings.
 * Permite consultas de hash de s[L...R] em O(1).
 */
struct StringHash {
    string s;
    int n;
    ll base;
    ll mod;
    vector<ll> prefix_hash; // prefix_hash[i] = hash de s[0...i-1]
    vector<ll> powers;      // powers[i] = (base^i) % mod

    StringHash(const string& str, ll b = 31, ll m = 1e9 + 9) {
        s = str;
        n = s.length();
        base = b;
        mod = m;
        
        prefix_hash.resize(n + 1);
        powers.resize(n + 1);

        powers[0] = 1;
        prefix_hash[0] = 0;

        for (int i = 1; i <= n; i++) {
            // (base^i) % mod
            powers[i] = (powers[i - 1] * base) % mod;
            
            // hash(s[0...i-1]) = (hash(s[0...i-2]) * base + s[i-1]) % mod
            // s[i-1] é o caractere na posição i (0-based)
            prefix_hash[i] = (prefix_hash[i - 1] * base + (s[i - 1] - 'a' + 1)) % mod;
            // (Use (s[i-1] - 'A' + 1) para maiúsculas,
            //  ou apenas s[i-1] se os caracteres puderem ser 0)
        }
    }

    /**
     * @brief Pega o hash da substring s[L...R] (inclusivo, 0-based).
     * @param L Índice inicial.
     * @param R Índice final.
     * @return O valor do hash.
     */
    ll getHash(int L, int R) {
        int len = R - L + 1;
        
        // hash(s[L..R]) = hash(s[0..R]) - hash(s[0..L-1]) * (base^len)
        ll hash_R = prefix_hash[R + 1];
        ll hash_L = prefix_hash[L];
        
        ll hash_L_shifted = (hash_L * powers[len]) % mod;
        
        // (hash_R - hash_L_shifted + mod) % mod (para lidar com subtração negativa)
        return (hash_R - hash_L_shifted + mod) % mod;
    }
};

/*
// Exemplo de uso:
string texto = "abacaba";
StringHash hasher(texto);

// hash("aba") em s[0..2]
cout << hasher.getHash(0, 2) << endl; 

// hash("aba") em s[4..6]
cout << hasher.getHash(4, 6) << endl; // Deve ser igual ao anterior
*/
```

-----

### 1\. Problema Clássico: Maior Soma de Subarray Contíguo (Algoritmo de Kadane)

Este é, de longe, o problema mais famoso relacionado a "sub soma contígua". O objetivo é encontrar o subarray contíguo (com pelo menos um elemento) que possui a maior soma.

O Algoritmo de Kadane resolve isso de forma genial em tempo $O(N)$.

```cpp
#include <iostream>
#include <vector>
#include <algorithm> // Para std::max

#define ll long long
using namespace std;

/**
 * @brief Encontra a maior soma de um subarray contíguo.
 * (Algoritmo de Kadane)
 *
 * @param arr O vetor de entrada.
 * @return A maior soma possível.
 */
ll kadane(const vector<int>& arr) {
    if (arr.empty()) {
        return 0; // Ou algum valor de erro
    }

    ll max_global = arr[0]; // A maior soma encontrada até agora
    ll max_atual = arr[0];  // A maior soma terminando nesta posição

    // Começa do segundo elemento
    for (size_t i = 1; i < arr.size(); ++i) {
        // Para a soma máxima terminando em 'i', temos duas escolhas:
        // 1. Começar um novo subarray em 'arr[i]'.
        // 2. Anexar 'arr[i]' ao subarray anterior.
        max_atual = max((ll)arr[i], max_atual + arr[i]);
        
        // A maior soma global é o máximo de todas as 'max_atual'
        max_global = max(max_global, max_atual);
    }

    return max_global;
}

/*
// Exemplo de uso:
int main() {
    vector<int> v = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    // O subarray com a maior soma é [4, -1, 2, 1], soma = 6
    cout << "Maior soma: " << kadane(v) << endl; // Output: 6

    vector<int> v_neg = {-5, -1, -3};
    // Se todos são negativos, a resposta é o maior (menos negativo)
    cout << "Maior soma: " << kadane(v_neg) << endl; // Output: -1
    return 0;
}
*/
```

-----

### 2\. Problema da Soma Alvo: Contar Subarrays com Soma Exata `k` (Somas de Prefixo + Map)

Este é outro problema muito comum. "Quantos subarrays contíguos somam exatamente `k`?"

A técnica usa Somas de Prefixo e um `map` (ou `unordered_map`) para resolver em tempo $O(N)$ (em média) ou $O(N \log N)$ (com `map`).

**A Lógica:**

1.  A soma de um subarray `arr[i...j]` é `soma_prefixo[j] - soma_prefixo[i-1]`.
2.  Estamos procurando por `soma_prefixo[j] - soma_prefixo[i-1] == k`.
3.  Reorganizando: `soma_prefixo[i-1] == soma_prefixo[j] - k`.
4.  Enquanto iteramos pelo array (calculando `soma_prefixo[j]`), nós procuramos no `map` quantas vezes já vimos o valor `soma_prefixo[j] - k`.

<!-- end list -->

```cpp
#include <iostream>
#include <vector>
#include <map> // Ou <unordered_map> para O(N) em média

#define ll long long
using namespace std;

/**
 * @brief Conta o número de subarrays contíguos cuja soma é igual a 'k'.
 *
 * @param arr O vetor de entrada.
 * @param k A soma alvo.
 * @return O número de subarrays.
 */
int countSubarraysWithSum(const vector<int>& arr, int k) {
    int n = arr.size();
    if (n == 0) return 0;

    // map<soma_prefixo, contagem>
    // Armazena a frequência de cada soma de prefixo encontrada.
    map<ll, int> prefix_sum_counts;
    
    ll current_sum = 0;
    int total_count = 0;

    // Caso base: uma soma de 0 existe (o prefixo vazio)
    // Isso é crucial para contar subarrays que começam no índice 0
    prefix_sum_counts[0] = 1;

    for (int x : arr) {
        // current_sum é a soma_prefixo[j]
        current_sum += x;
        
        // Estamos procurando por (current_sum - k)
        ll complement = current_sum - k;

        // Se encontramos o complemento no map, todos os prefixos
        // que terminaram com 'complement' formam um subarray válido
        // com o 'current_sum' atual.
        if (prefix_sum_counts.count(complement)) {
            total_count += prefix_sum_counts[complement];
        }

        // Adiciona a soma de prefixo atual ao map
        prefix_sum_counts[current_sum]++;
    }

    return total_count;
}

/*
// Exemplo de uso:
int main() {
    vector<int> v = {1, 2, 3, 0, -1, 5};
    int k = 5;
    // Subarrays:
    // [2, 3]
    // [2, 3, 0]
    // [0, -1, 5, 1] (brincadeira, {0, -1, 5} não, {5})
    // [1, 2, 3, 0, -1]
    // [5]
    // Corrigindo:
    // [2, 3] (soma 5)
    // [2, 3, 0] (soma 5)
    // [5] (soma 5)
    vector<int> v2 = {1, 1, 1};
    k = 2;
    // [1, 1] (índice 0, 1)
    // [1, 1] (índice 1, 2)
    
    cout << "Contagem: " << countSubarraysWithSum(v2, k) << endl; // Output: 2

    vector<int> v3 = {3, 4, 7, 2, -3, 1, 4, 2};
    k = 7;
    // [3, 4]
    // [7]
    // [7, 2, -3, 1]
    // [4, 2, -3, 1, 4] (não)
    // [2, -3, 1, 4, 2] (não)
    // [-3, 1, 4, 2] (não)
    // [1, 4, 2]
    cout << "Contagem: " << countSubarraysWithSum(v3, 7) << endl; // Output: 4
    return 0;
}
*/
```

**Nota:** Para apenas *verificar a existência* (Cenário 2), você pode modificar a função acima para retornar `true` assim que `total_count` for maior que 0, ou simplesmente checar se `countSubarraysWithSum(arr, k) > 0`.

-----

### 3\. Problema da Soma Alvo (Variante): Janela Deslizante (Sliding Window)

Se você tem a **garantia** de que todos os números no vetor são **não-negativos (\>= 0)**, você pode usar uma técnica mais simples e eficiente (em termos de memória) chamada "Janela Deslizante" (Sliding Window / Two Pointers).

```cpp
#include <iostream>
#include <vector>

#define ll long long
using namespace std;

/**
 * @brief Encontra se existe um subarray contíguo com soma 'k'.
 * REQUISITO: Todos os elementos em 'arr' devem ser NÃO-NEGATIVOS.
 *
 * @param arr O vetor (apenas números >= 0).
 * @param k A soma alvo.
 * @return true se tal subarray existe, false caso contrário.
 */
bool findSubarraySum_SlidingWindow(const vector<int>& arr, int k) {
    int n = arr.size();
    ll current_sum = 0;
    int left = 0; // O início da janela

    // 'right' é o fim da janela
    for (int right = 0; right < n; ++right) {
        // 1. Expande a janela
        current_sum += arr[right];

        // 2. Contrai a janela pela esquerda até que a soma
        // seja <= k (ou a janela se torne inválida)
        while (current_sum > k && left <= right) {
            current_sum -= arr[left];
            left++;
        }

        // 3. Checa se a soma atual é o alvo
        if (current_sum == k) {
            // (O subarray é de 'left' até 'right')
            return true;
        }
    }

    return false;
}

/*
// Exemplo de uso:
int main() {
    vector<int> v = {1, 4, 20, 3, 10, 5};
    int k = 33;
    // Subarray [20, 3, 10] soma 33
    cout << "Existe soma 33? " << boolalpha 
         << findSubarraySum_SlidingWindow(v, k) << endl; // Output: true

    vector<int> v2 = {1, 4, 0, 0, 3, 10, 5};
    k = 7;
    // Subarray [4, 0, 0, 3] soma 7
    cout << "Existe soma 7? " << boolalpha 
         << findSubarraySum_SlidingWindow(v2, k) << endl; // Output: true
}
*/
```

-----

### 1\. `std::vector` (`<vector>`)

O "canivete suíço". É o seu array dinâmico padrão. Use-o em vez de arrays C-style (`int arr[]`) sempre que puder.

  * **Por que é útil?** Gerencia memória automaticamente, sabe o próprio tamanho, pode ser passado por referência para funções facilmente.
  * **Funções-chave:**
      * `v.push_back(x)`: Adiciona `x` ao final. $O(1)$ (amortizado).
      * `v.size()`: Retorna o número de elementos. $O(1)$.
      * `v[i]`: Acesso direto ao elemento `i`. $O(1)$.
      * `v.begin()`, `v.end()`: Iteradores para o início e o fim, usados por algoritmos.

<!-- end list -->

```cpp
#include <vector>
#include <iostream>

int main() {
    // Declara um vetor de inteiros
    std::vector<int> v;
    
    // Adiciona elementos
    v.push_back(10);
    v.push_back(20);
    v.push_back(30);

    // Itera sobre o vetor
    for (int i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " "; // Acesso por índice
    }
    std::cout << "\n";

    // "Range-based for loop" (C++11 em diante)
    for (int x : v) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    
    // Declara com tamanho 5, todo inicializado com -1
    std::vector<int> v_sized(5, -1); 
    
    return 0;
}
```

-----

### 2\. `std::priority_queue` (`<queue>`)

Uma fila de prioridade, implementada como um *heap*. Por padrão, é uma **max-heap** (o maior elemento está sempre no topo).

  * **Por que é útil?** Essencial para o Algoritmo de Dijkstra, simulações de eventos (processar o evento mais "urgente" primeiro), ou qualquer problema do tipo "pegue o maior/menor N itens".
  * **Funções-chave:**
      * `pq.push(x)`: Insere `x`. $O(\log N)$.
      * `pq.pop()`: Remove o elemento do topo. $O(\log N)$.
      * `pq.top()`: Retorna o elemento do topo (sem remover). $O(1)$.
      * `pq.empty()`: Retorna `true` se estiver vazia.

<!-- end list -->

```cpp
#include <queue>
#include <iostream>

int main() {
    // Max-heap (padrão)
    std::priority_queue<int> max_pq;
    max_pq.push(10);
    max_pq.push(30);
    max_pq.push(20);
    
    // Imprime 30
    std::cout << "Topo (max): " << max_pq.top() << std::endl; 

    // --- Como fazer uma Min-heap ---
    // (Pega o menor elemento primeiro)
    // A sintaxe é mais verbosa:
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_pq;
    min_pq.push(10);
    min_pq.push(30);
    min_pq.push(5);
    
    // Imprime 5
    std::cout << "Topo (min): " << min_pq.top() << std::endl; 
    
    // Truque comum para min-heap de inteiros:
    // Inserir os números negativos em uma max-heap
    std::priority_queue<int> fake_min_pq;
    fake_min_pq.push(-10); // negativo de 10
    fake_min_pq.push(-30); // negativo de 30
    fake_min_pq.push(-5);  // negativo de 5
    
    // Imprime -5. Para obter o valor original, pegue -top().
    std::cout << "Topo (fake min): " << -fake_min_pq.top() << std::endl; 
}
```

-----

### 3\. `std::set` e `std::map` (`<set>`, `<map>`)

Ambos são implementados como Árvores de Busca Binária Balanceadas (Red-Black Trees).

  * **`std::set`**: Armazena elementos **únicos** e **ordenados**.

  * **`std::map`**: Armazena pares **(chave, valor)**. As chaves são únicas e ordenadas.

  * **Por que são úteis?** Todas as operações (inserção, remoção, busca) são em $O(\log N)$.

      * `set`: Ótimo para saber se você "já viu" um elemento, ou para manter uma coleção de elementos sempre ordenada.
      * `map`: Perfeito para "contagem de frequência" ou para associar um valor a uma chave (ex: `map<string, int>`).

  * **Funções-chave (`set`):**

      * `s.insert(x)`: Insere `x`.
      * `s.erase(x)`: Remove `x`.
      * `s.find(x)`: Retorna um iterador para `x`, ou `s.end()` se não encontrar.
      * `s.count(x)`: Retorna 1 ou 0 (se `x` existe ou não).
      * `s.lower_bound(x)`: Encontra o primeiro elemento $\ge x$. $O(\log N)$.

  * **Funções-chave (`map`):**

      * `m[chave] = valor`: Acesso mais fácil. Se `chave` não existe, ela é criada\!
      * `m.find(chave)`: Busca pela chave.
      * `m.count(chave)`: Retorna 1 ou 0.

<!-- end list -->

```cpp
#include <map>
#include <set>
#include <iostream>
#include <string>

int main() {
    // --- Exemplo de Map (Contagem de Frequência) ---
    std::map<std::string, int> freq;
    freq["banana"]++;
    freq["abacaxi"]++;
    freq["banana"]++;

    // "banana" terá contagem 2
    std::cout << "bananas: " << freq["banana"] << std::endl; 

    // Itera sobre o map (em ordem alfabética de chave!)
    for (auto const& [chave, valor] : freq) {
        std::cout << chave << ": " << valor << std::endl;
    }
    // Output:
    // abacaxi: 1
    // banana: 2

    // --- Exemplo de Set ---
    std::set<int> s;
    s.insert(100);
    s.insert(50);
    s.insert(100); // Inserção ignorada (duplicata)

    // s contém {50, 100}
    if (s.count(50)) {
        std::cout << "50 existe" << std::endl;
    }
    
    // Imprime em ordem
    for (int x : s) {
        std::cout << x << " "; // Output: 50 100
    }
    std::cout << "\n";
}
```

**Observação:** Existem `std::unordered_set` e `std::unordered_map` (baseados em Hashing). São $O(1)$ em média, mas podem sofrer com colisões (como discutimos) e levar a $O(N)$ no pior caso. Use `map` e `set` se $O(\log N)$ for aceitável; use os `unordered` se você *realmente* precisar de $O(1)$ e os testes não forem maliciosos.

-----

### 4\. Algoritmos (`<algorithm>`)

Este *header* é um tesouro.

  * **`std::sort(v.begin(), v.end())`**: Ordena o `vector v`. $O(N \log N)$.
  * **`std::lower_bound(v.begin(), v.end(), x)`**: Em um `vector` **ordenado**, encontra o iterador para o primeiro elemento $\ge x$. $O(\log N)$.
  * **`std::upper_bound(v.begin(), v.end(), x)`**: Em um `vector` **ordenado**, encontra o iterador para o primeiro elemento $> x$. $O(\log N)$.
  * **`std::binary_search(v.begin(), v.end(), x)`**: Retorna `true`/`false` se `x` existe no `vector` **ordenado**.
  * **`std::next_permutation(v.begin(), v.end())`**: Reorganiza `v` para a próxima permutação lexicográfica. Útil para "força-bruta" em problemas com $N \le 10$.
  * **`std::reverse(v.begin(), v.end())`**: Reverte o vetor.
  * **`std::max_element(v.begin(), v.end())`**: Retorna um iterador para o maior elemento.
  * **`std::min_element(v.begin(), v.end())`**: Retorna um iterador para o menor elemento.

<!-- end list -->

```cpp
#include <algorithm>
#include <vector>
#include <iostream>

// Função de comparação customizada para o sort
bool comparadorCustomizado(int a, int b) {
    // Ordena por maior primeiro
    return a > b;
}

int main() {
    std::vector<int> v = {10, 50, 20, 40, 30};
    
    // Ordenação padrão (crescente)
    std::sort(v.begin(), v.end());
    // v agora é {10, 20, 30, 40, 50}

    // Busca binária
    if (std::binary_search(v.begin(), v.end(), 40)) {
        std::cout << "40 foi encontrado!" << std::endl;
    }

    // Encontra o primeiro elemento >= 25
    auto it = std::lower_bound(v.begin(), v.end(), 25);
    // *it será 30
    std::cout << "Lower bound de 25: " << *it << std::endl; 

    // Ordenação com comparador customizado
    std::sort(v.begin(), v.end(), comparadorCustomizado);
    // v agora é {50, 40, 30, 20, 10}
    for (int x : v) std::cout << x << " ";
    std::cout << "\n";
    
    // Gera permutações
    std::string s = "abc";
    do {
        std::cout << s << std::endl;
    } while (std::next_permutation(s.begin(), s.end()));
    // Output:
    // abc
    // acb
    // bac
    // bca
    // cab
    // cba
}
```

-----

### 5\. `std::pair` (`<utility>`)

Simplesmente um contêiner que agrupa dois valores.

  * **Por que é útil?** Usado em todos os lugares: `map<int, int>`, `priority_queue<pair<int, int>>` (para Dijkstra, guardando {distância, nó}), `vector<pair<int, int>>` (para guardar coordenadas {x, y}).
  * **Como usar:**
      * `pair<int, string> p;`
      * `p.first = 10;`
      * `p.second = "hello";`
      * `p = make_pair(10, "hello");` (estilo antigo)
      * `p = {10, "hello"};` (estilo moderno)



Em C++, a lista de adjacência (`adj`) é comumente representada por um "vetor de vetores".

```cpp
vector<vector<int>> adj; 
```

Nessa estrutura, `adj[u]` é um vetor que armazena todos os vizinhos `v` do nó `u`.

O formato de entrada mais comum em competições é:

  * **Linha 1:** `N` (número de nós) e `M` (número de arestas).
  * **Próximas `M` linhas:** `u v` (indicando uma aresta entre o nó `u` e o nó `v`).

### O Ponto Crucial: Indexação 0 vs. 1

1.  **Problemas de Competição:** Geralmente usam **indexação 1** (nós numerados de 1 a `N`).
2.  **Vetores em C++:** Sempre usam **indexação 0** (índices de 0 a `N-1`).

Você tem duas maneiras de lidar com isso:

-----

### Opção 1 (Recomendada): Usar indexação 1 (Redimensionar para `N + 1`)

Esta é a abordagem mais segura e rápida de codar, pois você não precisa fazer contas de +1/-1. Você simplesmente redimensiona seu vetor `adj` para o tamanho `N + 1` e ignora o índice 0.

**Exemplo: Grafo NÃO-DIRECIONADO (para BFS, DFS)**
(Se a aresta `u v` existe, a aresta `v u` também existe)

```cpp
#include <vector>
#include <iostream>

using namespace std;

int main() {
    int N, M; // N = nós, M = arestas
    
    // Suponha que a entrada seja:
    // 4 3        (N=4 nós, M=3 arestas)
    // 1 2        (Aresta entre 1 e 2)
    // 1 3        (Aresta entre 1 e 3)
    // 2 4        (Aresta entre 2 e 4)
    
    cin >> N >> M;

    // Redimensiona 'adj' para ter N+1 posições (índices 0, 1, 2, ..., N)
    // Vamos usar apenas os índices [1, N]
    vector<vector<int>> adj(N + 1); 

    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v; // Ex: lê 1 e 2

        // Adiciona a aresta u -> v
        adj[u].push_back(v); // adj[1] agora contém [2]

        // Como é NÃO-DIRECIONADO, adiciona também a aresta v -> u
        adj[v].push_back(u); // adj[2] agora contém [1]
    }

    // Ao final do loop:
    // adj[0] = [] (vazio)
    // adj[1] = [2, 3]
    // adj[2] = [1, 4]
    // adj[3] = [1]
    // adj[4] = [2]
    
    // Agora 'adj' está pronta para ser usada:
    // bfs(1); // Inicia a busca a partir do nó 1
    
    return 0;
}
```

-----

### Opção 2 (Alternativa): Converter para indexação 0 (Redimensionar para `N`)

Você pode economizar um pouquinho de memória redimensionando para `N` e subtraindo 1 de cada nó lido.

```cpp
int N, M;
cin >> N >> M;

// Redimensiona 'adj' para ter N posições (índices 0, 1, ..., N-1)
vector<vector<int>> adj(N); 

for (int i = 0; i < M; ++i) {
    int u, v;
    cin >> u >> v; // Ex: lê 1 e 2

    // Converte para 0-indexado
    u--; // u agora é 0
    v--; // v agora é 1

    adj[u].push_back(v); // adj[0] agora contém [1]
    adj[v].push_back(u); // adj[1] agora contém [0]
}

// Ao final do loop (para a mesma entrada):
// adj[0] = [1, 2]  (nó 1)
// adj[1] = [0, 3]  (nó 2)
// adj[2] = [0]     (nó 3)
// adj[3] = [1]     (nó 4)

// Lembre-se de chamar seus algoritmos com o índice 0:
// bfs(0);
```

-----

### Variações Importantes

1.  **Grafos Direcionados (ex: Ordenação Topológica):**
    Se a aresta é $u \to v$ (uma seta), você **não** adiciona a aresta de volta. Você faria *apenas*:

    ```cpp
    int u, v;
    cin >> u >> v;
    // (u-- e v-- se for 0-indexado)
    adj[u].push_back(v); 
    ```

2.  **Grafos com Pesos (ex: Dijkstra):**
    A entrada será `u v w` (w = peso). Sua estrutura `adj` muda para armazenar pares `{vizinho, peso}`.

    ```cpp
    // (pair<vizinho, peso>)
    vector<vector<pair<int, int>>> adj_ponderada(N + 1);

    for (int i = 0; i < M; ++i) {
        int u, v, w;
        cin >> u >> v >> w; // Lê origem, destino, peso

        adj_ponderada[u].push_back( {v, w} );
        // ou: adj_ponderada[u].push_back(make_pair(v, w));

        // Se for não-direcionado, adicione a volta:
        // adj_ponderada[v].push_back( {u, w} );
    }
    ```

    Isso preencheria a lista de adjacência adj que você viu no exemplo do Dijkstra (vector<map<int, int>> ou vector<vector<pair<int, int>>>).
