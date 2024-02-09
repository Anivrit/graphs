// Include necessary libraries
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include <queue>
#include <unordered_map>

// Use the standard namespace
using namespace std; 

// Node class definition
class Node {
public:
    int id;  // Node ID

    // Mapping a neighbor to edge weight
    std::unordered_map<Node*, int> neighbors;

    // Node constructor
    Node(int _id) : id(_id) {}

    // Method to store an edge. We can store bad edges with weight of -1. 
    // This kind of complicates code understanding, but makes code less bloated. 
    void storeEdge(Node* destination, int weight) {
        neighbors[destination] = weight;
    }
};

// FlowGraph class definition
class FlowGraph {
public:
    // Need this hash because there is no default hash for pairs. 
    struct pair_hash {
        template <class T1, class T2>
        std::size_t operator () (const std::pair<T1,T2> &p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second); 

            // Mainly for demonstration purposes, i.e. works but is overly simple
            // In the real world, use sth. like boost.hash_combine
            return h1 ^ h2;  
        }
    };

    // Map to store flow values
    std::unordered_map<std::pair<Node*, Node*>, int, pair_hash> flowValues;

    // Method to set the flow value for a specific edge. Returns true if the flow is within capacity. 
    bool setFlow(Node* source, Node* destination, int flow) {
        if (source->neighbors.find(destination) != source->neighbors.end()) {
            int edgeCapacity = source->neighbors[destination];
            if (flow <= edgeCapacity) {
                flowValues[std::make_pair(source, destination)] = flow;
                return true;
            }
            else{
                cout << "Flow exceeds edge capacity" << endl;
                flowValues[std::make_pair(source, destination)] = flow;
            }
        }
        else{
            cout << "Edge does not exist" << endl;
        }
        return false;
    }

    // Method to get the flow value for a specific edge
    int getFlow(Node* source, Node* destination) const {
        auto it = flowValues.find(std::make_pair(source, destination));
        return (it != flowValues.end()) ? it->second : 0; // Default to 0 if not found
    }

    // Method to calculate the total flow from a source node
    int calculateFlow(Node* source){
        int totalFlow = 0;
         for (const auto& neighborPair : source->neighbors) {
            Node* neighbor = neighborPair.first;
            totalFlow += flowValues[make_pair(source,neighbor)];
            cout  << source->id << " " << neighbor->id << " " << flowValues[make_pair(source,neighbor)] << endl;
         }
        return totalFlow;
    }
};

// Graph class definition
class Graph {

const int INF = 1e9; // 1 billion

public:
    std::vector<Node*> nodes;  // Vector to store nodes

    // Method to add a node to the graph
    virtual void addNode(Node* node) {
        nodes.push_back(node);
    }

    // Abstract method to add an edge between two nodes (either good or bad)
    virtual void addEdge(int _weight, Node* _source, Node* _destination)  =0;

    // Method to print the path
    virtual void printPath(vector<Node*> final_path){
        if(final_path.empty()){
            cout << "Path Does Not Exist" << endl;
        }
        else{
            cout << "Path: " << endl;
            for (Node* pathNode : final_path){
                cout << pathNode->id << " ";
            }
            cout << "\nPath Length: " << final_path.size() << endl;
        }
    }

    // Method to perform Breadth-First Search (BFS) on the graph
    virtual vector<Node*> bfsGraph(Node* start, Node* destination, FlowGraph* f = nullptr) {
        unordered_set<int> visited_nodes;  // Set to store visited nodes
        queue<vector<Node*>> to_visit;  // Queue to store nodes to visit
        to_visit.push(vector<Node*>{start});
        vector<Node*> final_path;  // Vector to store the final path

        while (!to_visit.empty()) {
            vector<Node*> path = to_visit.front();
            to_visit.pop();
            int check_id = path.back()->id;

            if (check_id == destination->id) {
                final_path = path;
                break;
            } else {
                for (const auto& neighborPair : path.back()->neighbors) {
                    Node* neighbor = neighborPair.first;
                    if (visited_nodes.count(neighbor->id) == 0) {
                        // Create a new vector for each neighbor
                        if(f != nullptr){
                            int flow = f->getFlow(path.back(),neighbor);
                            int capacity = path.back()->neighbors[neighbor];
                            if(flow >= capacity){
                                continue; //Will skip the code that comes after. 
                            }
                        }
                        vector<Node*> new_path = path;
                        new_path.push_back(neighbor);
                        to_visit.push(new_path);
                        visited_nodes.insert(neighbor->id);
                    }
                }
            }
        }
        
        return final_path;
    }

    // Method to perform Depth-First Search (DFS) on the graph
    virtual vector<Node*> dfsGraph(Node* current, Node* destination) {
        std::vector<Node*> path;  // Vector to store the path
        std::unordered_set<int> visited_nodes;  // Set to store visited nodes
        std::vector<Node*> final_path;  // Vector to store the final path

        return dfsGraphRecursive(current, destination, path, visited_nodes, final_path);
    }

    // Recursive method to perform DFS
    virtual vector<Node*> dfsGraphRecursive(Node* current, Node* destination, std::vector<Node*>& path,
                       std::unordered_set<int>& visited_nodes, std::vector<Node*>& final_path) {
        path.push_back(current);

        // Mark the current node as visited
        visited_nodes.insert(current->id);

        // Check if the current node is the target
        if (current->id == destination->id) {
            final_path = path;
            return final_path;
        }

        // Recursively explore neighbors
        for (const auto& neighborPair : current->neighbors) {
            Node* neighbor = neighborPair.first;
            if (visited_nodes.count(neighbor->id) == 0) {
                return dfsGraphRecursive(neighbor, destination, path, visited_nodes, final_path);
            }
        }
        // Backtrack by removing the current node from the path
        path.pop_back();
        return path;

    }

    // Method to print the graph
    virtual void printGraph() {
        std::cout << "Nodes:" << std::endl;
        for (Node* node : nodes) {
            std::cout << "Node " << node->id << std::endl;
            std::cout << "Neighbors: " << std::endl;
            for (const auto& neighborPair : node->neighbors) {
                Node* neighbor = neighborPair.first;
                int weight = neighborPair.second;
                cout << "Node: " << neighbor->id << " Weight: " << weight << std::endl;
            }
            std::cout << std::endl;
        }
    }

    // Method to perform Djikstra's algorithm
    virtual vector<int> djikstra(Node* source){
        int num_vertices = nodes.size(); 
        int source_id = source->id;
        //Syntax to generate a heap from priority queue
        // The second arg is the container used to store elements. Third is comparison operator for heap/pq.
        priority_queue<pair<int,Node*>, vector<pair<int,Node*>>, greater<pair<int,Node*>> > pq;

        //Distances from source stored here. Nodes need to be zero indexed! 
        vector<int> distances (num_vertices,INF);
    

        //pairs store distance, node
        pq.push(make_pair(0,source));
        distances[source_id] = 0;

        while(!pq.empty()){
            Node* top = pq.top().second;
            pq.pop();
            for (const auto& neighborPair : top->neighbors) {
                Node* neighbor = neighborPair.first;
                int weight = neighborPair.second;
                if(distances[neighbor->id] > distances[top->id] + weight){
                    distances[neighbor->id] = distances[top->id] + weight;
                    pq.push(make_pair(distances[neighbor->id],neighbor));
                }
            }
        }
        return distances;
    }

    // Method to perform Bellman-Ford algorithm
    virtual vector<int> bellFord(Node* source){
        int num_vertices = nodes.size();
        int num_edges = 0;
        for (const auto& node : nodes) {
            num_edges += node->neighbors.size();
        }
        vector<int> distances (num_vertices,INF);
        distances[source->id] = 0;
        //Edge relaxation like in Djikstra. But in Djikstra performed only on nodes in pq, here do it on every edge. And do it V-1 times (since max path is V-1 without loop)

        for(int i = 0; i< num_vertices-1; i++){
            for (const auto& node : nodes) {
                for (const auto& neighborPair : node->neighbors) {
                    Node* neighbor = neighborPair.first;
                    int weight = neighborPair.second;
                    if(distances[neighbor->id] > distances[node->id] + weight){
                        distances[neighbor->id] = distances[node->id] + weight;
                    }
                }
            }
        }

        // One last relaxation. If distance decreases, negative cycle detected, return exception. 
        for (const auto& node : nodes) {
            for (const auto& neighborPair : node->neighbors) {
                Node* neighbor = neighborPair.first;
                int weight = neighborPair.second;
                if(distances[neighbor->id] > distances[node->id] + weight){
                    return {-1};
                }
            }
        }
        return distances; 
    }
    
    // Method to perform Ford-Fulkerson algorithm
    virtual int fordFulkerson(Node* source, Node*destination){
        FlowGraph f;
        vector<Node*> path = bfsGraph(source,destination,&f);
        while(!path.empty()){
            int n = path.size();
            int flow = INF;
            for(int i =0; i<n-1; i++){
                int capacity = path[i]->neighbors[path[i+1]];
                flow = max(min(flow,capacity-f.getFlow(path[i],path[i+1])),0);

            }
            
            for(int i =0; i<n-1; i++){
                //NEED TO CHECK: if bfs Graph and setFlow are double checking the flow <= capacity condition. 
                f.setFlow(path[i],path[i+1],f.getFlow(path[i],path[i+1]) + flow);

            }
            path = bfsGraph(source,destination,&f);
            
            int n2 = path.size();
        }   
        
        return f.calculateFlow(source);
    }
    
};

// UndirectedGraph class definition
class UndirectedGraph : public Graph {

const int INF = 1e9; // 1 billion

public:
    UndirectedGraph() {}

    // Method to add an edge in an undirected graph
    void addEdge(int _weight, Node* _source, Node* _destination){
        _source->storeEdge(_destination,_weight);
        _destination->storeEdge(_source,_weight);
    }

    // Method to perform recursion in the badEdgePath method
    int recurse(Node* source, Node* destination, int numBadUsed, int totalUsable, int numToTraverse,vector<vector<int>>& dp) {
        
        //There is a loophole in the logic. We can be caught in an infinite loop, where recurse calls a neighboring edge, whose recurse
        //function calls the previous edge again, and so on. We can simply fix this by storing a reference to a set of visited nodes, but
        // I just decided to store the number of nodes traversed as it was a bit cleaner. If you have traversed more than n-1 edges, you
        // cannot be on the shortest path. 
        if (numToTraverse < 0){
            return INF;
        }

        // Base case: Reached the destination
        if ((source->id == destination->id)) {
            return 0;
        }

        // Memoization: Check if the result is already computed
        if (dp[numBadUsed][source->id] != INF) {
            return dp[numBadUsed][source->id];
        }

        // Explore neighbors
        int minDist = INF;
        for (const auto& neighborPair : source->neighbors) {
            Node* neighbor = neighborPair.first;
            int weight = neighborPair.second;

            if (weight < 0 && numBadUsed >= totalUsable) {
                continue; // Skip bad edge if not allowed
            }

            int updatedNumBadUsed = numBadUsed + (weight < 0 ? 1 : 0);
            int weightWithSign = (weight < 0) ? -weight : weight;

            if (dp[updatedNumBadUsed][neighbor->id] == INF) {
                minDist = min(minDist, weightWithSign + recurse(neighbor, destination, updatedNumBadUsed, totalUsable,numToTraverse-1, dp));
            } else {
                minDist = min(minDist, weightWithSign + dp[updatedNumBadUsed][neighbor->id]);
            }
        }

        // Memoize the result
        dp[numBadUsed][source->id] = minDist;
        return minDist;
    }

    // Function to calculate the shortest path between source and destination, where you can use at most numBad bad edges.
    int badEdgePath(Node* source, Node* destination, int numBad) {
        int numNodes = nodes.size();
        vector<vector<int>> dp(numBad + 1, vector<int>(numNodes, INF));

        // Call the recursive function
        int result = recurse(source, destination, 0, numBad, numNodes-1,dp);

        return (result == INF) ? -1 : result; // Return -1 if there is no valid path
    }

};

// DirectedGraph class definition
class DirectedGraph : public Graph {

public:
    DirectedGraph() {}

    // Method to add an edge in a directed graph
    void addEdge(int _weight, Node* _source, Node* _destination){
        _source->storeEdge(_destination,_weight);
    }
};

// Main function
int main() {
    // Create a graph
    UndirectedGraph graph;

    // Create nodes. Must be zero indexed. 
    Node* node0 = new Node(0);
    Node* node1 = new Node(1);
    Node* node2 = new Node(2);
    Node* node3 = new Node(3);
    Node* node4 = new Node(4);
    Node* node5 = new Node(5);
    Node* node6 = new Node(6);

    // Add nodes to the graph
    graph.addNode(node0);
    graph.addNode(node1);
    graph.addNode(node2);
    graph.addNode(node3);
    graph.addNode(node4);
    graph.addNode(node5);
    graph.addNode(node6);

    // Define good and bad edges using the addEdge function
    graph.addEdge(5,node1, node2);
    graph.addEdge(1,node2, node3);
    graph.addEdge(1,node2, node5);
    graph.addEdge(6,node3, node4);
    graph.addEdge(1,node4, node5);
    graph.addEdge(3,node5, node6);
    graph.addEdge(4,node6, node1);
    graph.addEdge(6,node0,node2);

    graph.addEdge(2,node1, node3);
    graph.addEdge(5,node3, node5);

    // Print information about the graph
    graph.printGraph();

    graph.printPath(graph.bfsGraph(node1,node4));
    graph.printPath(graph.dfsGraph(node1,node5));

    cout << endl << "Ford Fulkerson MaxFlow/MinCut" << endl;

    int maxFlow = graph.fordFulkerson(node1,node5);
    cout << "Max Flow / Min Cut Amount: " << maxFlow << endl;
    
    cout << endl << "BellMan Ford on Node 1" << endl;
    vector<int> resultBF = graph.bellFord(node1);
    auto iter2 = resultBF.begin();
    bool hasNegativeCycle; 
    if(*iter2 == -1)
    {
        cout << "Negative Cycle Detected" << endl;    
        hasNegativeCycle = true;
    } 
    else{
        for (; iter2 != resultBF.end(); ++iter2) {
            std::cout << *iter2 << " ";
        }
    }
    if(hasNegativeCycle != true){
        cout << endl << "Djikstra's on Node 1" << endl;
        vector<int> resultDjikstra = graph.djikstra(node1);
        auto iter = resultDjikstra.begin();
        for (; iter != resultDjikstra.end(); ++iter) {
            std::cout << *iter << " ";
        }
    }

        // Clean up memory (deallocate nodes)
    for (Node* node : graph.nodes) {
        delete node;
    }
    
    return 0;
}
