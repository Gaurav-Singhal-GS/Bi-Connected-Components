/*
 ============================================================================
 Name        : binconnected_components.cu
 Author      : Diptanshu, Gaurav
 Version     :
 Copyright   : (c) 2018
 Description : CUDA compute reciprocals
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SWAP(a, b) {int swp=a; a=b; b=swp;}
#define MAX_HEIGHT 10

int numIterations;

/*
 * */
__global__ void bfs(int *adjList, int *offset, int *inpFrontier, int *outFrontier,
					int *parent, int *visited, int *treeEdges, int *s1, int *s2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < *s1 && visited[inpFrontier[tid]] == 0) {
    	int v = inpFrontier[tid];	// Current vertex
    	// Put all the unvisited neighbors into outFrontier
    	for (int i = offset[v]; i < offset[v + 1]; i++) {
    		if (!visited[adjList[i]] && atomicCAS(&parent[adjList[i]], -1, v) == -1) {
				int old = atomicAdd(s2, 1);
				outFrontier[old] = adjList[i];
				treeEdges[i] = -1;
    		}
    		else if (adjList[i] == parent[v]) {
				treeEdges[i] = -2;
				// Place the parent as the first element in adjList
				if (i != offset[v]) {
					SWAP(adjList[offset[v]], adjList[i]);
					SWAP(treeEdges[offset[v]], treeEdges[i]);
				}
			}
    		else if (v < adjList[i]) {
    			// Non tree edge, mark only in one direction such that a < b for any non-tree edge a->b.
    			treeEdges[i] = v;
    		}
    		else {
    			treeEdges[i] = -2;
    		}
    	}
    	visited[v] = 1;
    }
}

/*
 *
 * */
__global__ void lca(int *adjList, int *offset, int *parent, int *nonTreeEdges,
                    int *unfinished, int *threadEdge, int *lcaThread, int *auxAdjList,
                    int vertexCount)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x, i = 0, len1, len2;
	int a = nonTreeEdges[3 * tid];
	int b = nonTreeEdges[3 * tid + 1];
	int eid = nonTreeEdges[3 * tid + 2];
	int path_a[MAX_HEIGHT], path_b[MAX_HEIGHT];

	while (a != 0)
	{
		path_a[i++] = a;
		a = parent[a];
	}
	path_a[i++] = 0;
	len1 = i;
	i = 0;

	while (b != 0)
	{
		path_b[i++] = b;
		b = parent[b];
	}
	path_b[i++] = 0;
	len2 = i;

	i = 0;
	while (i < len1 && i < len2 && path_a[len1 - i - 1] == path_b[len2 - i - 1])
		i++;

	int lcaVertex = path_a[len1 - i];
	//printf("Edge %d: %d %d LCA %d\n", eid, nonTreeEdges[3 * tid], nonTreeEdges[3 * tid + 1], lcaVertex);


	len1 -= i;
	len2 -= i;
	lcaThread[tid] = lcaVertex;


	// Mark the non-tree edge visited
	threadEdge[eid] = tid;

	// Mark the rest of the edges visited and the vertices as part of unfinished traversal
	for (i = 0; i < len1; i++) {
		threadEdge[offset[path_a[i]]] = tid;
		if (i != len1 - 1)
			unfinished[path_a[i]] = 1;
	}


	for (i = 0; i < len2; i++) {
		threadEdge[offset[path_b[i]]] = tid;
		if (i != len2 - 1)
			unfinished[path_b[i]] = 1;
	}

	__syncthreads();

	// Create auxiliary vertex
	// Special case for root vertex
	// As root vertex doesn't have any parent, we don't set its parent.
	if (lcaVertex != 0)
		auxAdjList[2 * lcaVertex] = adjList[offset[lcaVertex]];
	auxAdjList[2 * lcaVertex + 1] = lcaVertex;

}

__global__ void lca1(int *adjList, int *offset, int *parent, int *nonTreeEdges,
                    int *unfinished, int *threadEdge, int *lcaThread, int *auxAdjList,
                    int vertexCount, int edgeCount)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x, i = 0, len1, len2;
	int a = nonTreeEdges[3 * tid];
	int b = nonTreeEdges[3 * tid + 1];
	if (auxAdjList[2 * a + 1] != -1)
		a += vertexCount;
	if (auxAdjList[2 * b + 1] != -1)
			b += vertexCount;

	int eid = nonTreeEdges[3 * tid + 2];
	int path_a[MAX_HEIGHT], path_b[MAX_HEIGHT];

	while (a != 0)
	{
		path_a[i++] = a;
		if (a < vertexCount && auxAdjList[2 * a + 1] != -1)
			a = vertexCount + a;
		else if (a >= vertexCount)
			a = parent[a - vertexCount];
		else
			a = parent[a];
	}
	path_a[i++] = 0;
	len1 = i;
	i = 0;

	while (b != 0)
	{
		path_b[i++] = b;
		if (b < vertexCount && auxAdjList[2 * b + 1] != -1)
			b = vertexCount + b;
		else if (b >= vertexCount)
			b = parent[b - vertexCount];
		else
			b = parent[b];
	}
	path_b[i++] = 0;
	len2 = i;

	i = 0;
	while (i < len1 && i < len2 && path_a[len1 - i - 1] == path_b[len2 - i - 1])
		i++;

	//int lcaVertex = path_a[len1 - i];
	//printf("Edge %d: %d %d LCA %d\n", eid, nonTreeEdges[3 * tid], nonTreeEdges[3 * tid + 1], lcaVertex);


	len1 -= i;
	len2 -= i;



	// Mark the non-tree edge visited
	threadEdge[eid] = tid;

	for (i = 0; i < len1; i++) {
		if (path_a[i] >= vertexCount) {
			threadEdge[edgeCount + path_a[i] - vertexCount] = tid;

		}
		else {
			threadEdge[offset[path_a[i]]] = tid;
		}
	}


	for (i = 0; i < len2; i++) {
		if (path_b[i] >= vertexCount) {
			threadEdge[edgeCount + path_b[i] - vertexCount] = tid;
		}
		else {
			threadEdge[offset[path_b[i]]] = tid;

		}
	}


}

__global__ void auxGraph(int *adjList, int *offset, int *lcaThread, int vertexCount,
		int *rootLCACount, int *auxAdjList) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int lcaVertex = lcaThread[tid];

	if (lcaVertex != 0)
		adjList[offset[lcaVertex]] = vertexCount + lcaVertex;
	else
		atomicAdd(rootLCACount, 1);

	// Update grandParent's child
	int grandParent = auxAdjList[2 * lcaVertex];
	for (int i = offset[grandParent]; i < offset[grandParent + 1]; i++) {
		if (adjList[i] == lcaVertex) {
			adjList[i] = vertexCount + lcaVertex;
			break;
		}
	}

}


__global__ void markArtPoint(int *adjList, int *offset, int *lcaThread, int *artPoint,
							int *unfinished, int *rootLCACount) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int lcaVertex = lcaThread[tid];
	bool bridge = false;


	for (int i = offset[lcaVertex]; i < offset[lcaVertex + 1]; i++) {
		if (!unfinished[adjList[i]]) {
			bridge = true;
			break;
		}
	}

	printf("vertex %d rootLCACOUnt %d bridge %d\n", lcaVertex, *rootLCACount, bridge);
	if (lcaVertex != 0 && bridge)
		artPoint[lcaVertex] = 1;
	else if (lcaVertex == 0 && bridge && *rootLCACount > 1)
		artPoint[0] = 1;
}
/*
 * Finds BCC Id for each edge. If an edge was part of the path to an LCA and
 * that LCA happens to be an articulation point, we assign the LCA's vertex ID as BCC id to the edge.
 * Otherwise, we traverse up the tree to find an LCA which is an articulation point.
 */
__global__ void findBCC(int *adjList, int *offset, int *threadEdge, int *lcaThread,
						int *artPoint, int *bccId) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int lcaVertex = threadEdge[tid];

	// TODO: Unfinished implementation
	// Note: For each undirected edge b/w a-b, only one direction is marked in the threadEdge
	if (lcaVertex != -1) {
		while (!artPoint[lcaVertex]) {
			//lcaVertex = adjList[offset[lcaVertex]];
		}

		bccId[tid] = lcaVertex;
	}

}

int main(int argc, char **argv)
{
	char* edgeListFile = argv[1];
	FILE *fp;
	fp = fopen(edgeListFile, "r");
	if (fp == NULL) {
		printf("ERROR: File does not exist!\n");
		return 1;
	}

	int vertexCount, edgeCount;
	fscanf(fp, "%d", &vertexCount);
	fscanf(fp, "%d", &edgeCount);
	printf("VertexCount %d\n", vertexCount);
	printf("EdgeCount %d\n", edgeCount);

	// Data structure to represent the graph in CSR format
	int *adjList;		// Concatenated adjacency list
	int *offset;		// Stores offset of each vertex's adjacency list

	size_t adjListSize = edgeCount * sizeof(int);
	size_t offsetSize = (vertexCount + 1) * sizeof(int);
	size_t verticesSize = vertexCount * sizeof(int);

	adjList = (int *)malloc(adjListSize);
	offset = (int *)malloc(offsetSize);

	int edgeCounter = 0, vertexCounter = 0;
	int prevSource, source, dest;
	fscanf(fp, "%d %d", &prevSource, &dest);

	// Convert the graph to CSR format
	while (edgeCounter != edgeCount) {
		while (vertexCounter <= prevSource)			// Includes the vertices with no edges
			offset[vertexCounter++] = edgeCounter;
		adjList[edgeCounter++] = dest;
		while (fscanf(fp, "%d %d", &source, &dest) == 2 && source == prevSource)
			adjList[edgeCounter++] = dest;
		prevSource = source;
	}

	// Mark the sentinel values so that the degree of any vertex i = offset[i + 1] - offset[i]
	while (vertexCounter <= vertexCount)
		offset[vertexCounter++] = edgeCount;


//	printf("Adjacency List\n");
//	for(int i = 0; i < edgeCount; i++) {
//		printf("%d ", adjList[i]);
//	}
//
//	printf("\nOffsets\n");
//	for(int i = 0; i < vertexCount + 1; i++) {
//		printf("%d ", offset[i]);
//	}
//	printf("\n");

	// Initialize other data structure to be used for bfs
	int *inpFrontier, *outFrontier, *visited, *parent, *treeEdges;
	int s1, s2;					// Size of input and output frontiers
	int treeEdgeCount = 0;

	inpFrontier = (int *)calloc(vertexCount, sizeof(int));
	outFrontier = (int *)calloc(vertexCount, sizeof(int));
	visited = (int *)calloc(vertexCount, sizeof(int));
	treeEdges = (int *)calloc(edgeCount, sizeof(int));
	parent = (int *)malloc(verticesSize);
	memset(parent, -1, verticesSize);
	s1 = 1; s2 = 0;
	inpFrontier[0] = 0;	// Inserting source vertex

	// Corresponding device data
	int *d_adjList, *d_offset;
	int *d_inpFrontier, *d_outFrontier, *d_visited, *d_parent, *d_treeEdges;
	int *d_s1, *d_s2;

	cudaMalloc(&d_adjList, adjListSize);
	cudaMalloc(&d_offset, offsetSize);
	cudaMalloc(&d_inpFrontier, verticesSize);
	cudaMalloc(&d_outFrontier, verticesSize);
	cudaMalloc(&d_visited, verticesSize);
	cudaMalloc(&d_treeEdges, edgeCount * sizeof(int));
	cudaMalloc(&d_parent, verticesSize);
	cudaMalloc(&d_s1, sizeof(int));
	cudaMalloc(&d_s2, sizeof(int));

	cudaMemcpy(d_adjList, adjList, adjListSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_offset, offset, offsetSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inpFrontier, inpFrontier, verticesSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_outFrontier, outFrontier, verticesSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_visited, visited, verticesSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_treeEdges, treeEdges, edgeCount * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_parent, parent, verticesSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_s1, &s1, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_s2, &s2, sizeof(int), cudaMemcpyHostToDevice);

	// Start the bfs
	bool odd = true;
	int inpQSize = s1;
	numIterations = 0;
	while (inpQSize != 0) {
		dim3 blocksPerGrid ((inpQSize + 1023) / 1024);
		dim3 threadsPerBlock ((inpQSize > 1024) ? 1024 : inpQSize);
		if (odd) {
			bfs<<<blocksPerGrid, threadsPerBlock>>>(d_adjList, d_offset, d_inpFrontier, d_outFrontier,
													d_parent, d_visited, d_treeEdges, d_s1, d_s2);
			cudaMemcpy(&inpQSize, d_s2, sizeof(int), cudaMemcpyDeviceToHost);
			s1 = 0;
			cudaMemcpy(d_s1, &s1, sizeof(int), cudaMemcpyHostToDevice);

		}
		else {
			bfs<<<blocksPerGrid, threadsPerBlock>>>(d_adjList, d_offset, d_outFrontier, d_inpFrontier,
													d_parent, d_visited, d_treeEdges, d_s2, d_s1);
			cudaMemcpy(&inpQSize, d_s1, sizeof(int), cudaMemcpyDeviceToHost);
			s2 = 0;
			cudaMemcpy(d_s2, &s2, sizeof(int), cudaMemcpyHostToDevice);
		}
		odd = !odd;
		numIterations++;
		treeEdgeCount += inpQSize;
	}

	cudaMemcpy(visited, d_visited, verticesSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(parent, d_parent, verticesSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(treeEdges, d_treeEdges, edgeCount * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(adjList, d_adjList, edgeCount * sizeof(int), cudaMemcpyDeviceToHost);



//	printf("Parent array\n");
//	for (int i = 0; i < vertexCount; i++)
//			printf("%d ", parent[i]);
//	printf("\n");
//
//	printf("Adjacency List\n");
//	for(int i = 0; i < edgeCount; i++) {
//		printf("(%d %d) ", i, adjList[i]);
//	}
//	printf("\n");
//
//	for (int i = 0; i < vertexCount; i++) {
//		if (parent[i] != adjList[offset[i]])
//			printf("WRONG %d\n", i);
//	}
//
//	printf("Number of iterations %d \n", numIterations);
//	printf("Visited array\n");
//	for (int i = 0; i < vertexCount; i++)
//		printf("%d ", visited[i]);
//	printf("\n");

//	printf("Tree Edges\n");
//	for (int i = 0; i < edgeCount; i++)
//		printf("%d ", treeEdges[i]);
//	printf("\n");

	int nonTreeEdgeCount = (edgeCount - 2 * treeEdgeCount) / 2;
//	printf("treeEdgecount %d\n", treeEdgeCount);
//	printf("Non-tree edges count %d\n", nonTreeEdgeCount);


	dim3 blocksPerGrid ((nonTreeEdgeCount + 1023) / 1024);
	dim3 threadsPerBlock ((nonTreeEdgeCount > 1024) ? 1024 : nonTreeEdgeCount);

	int threadCount = blocksPerGrid.x * threadsPerBlock.x;

	//printf("ThreadCount = %d\n", threadCount);

	// Data structure to represent non tree edges
	// a b i : edge a->b with edge id i
	int *nonTreeEdges = (int *) calloc(3 * nonTreeEdgeCount, sizeof(int));
	int *lcaThread = (int *) calloc(threadCount, sizeof(int));
	int *threadEdge = (int *) malloc(edgeCount * sizeof(int));
	memset(threadEdge, -1, edgeCount * sizeof(int));
	int *unfinished = (int *) calloc(vertexCount, sizeof(int));
	int *auxAdjList = (int *) malloc(2 * vertexCount * sizeof(int));
	memset(auxAdjList, -1, 2 * vertexCount * sizeof(int));
	int *artPoint = (int *) calloc(vertexCount, sizeof(int));
	int rootLCACount = 0;

	// Populate non tree edges
	for (int i = 0, j = 0; i < edgeCount; i++) {
		if (treeEdges[i] >= 0) {
			nonTreeEdges[j++] = treeEdges[i];
			nonTreeEdges[j++] = adjList[i];
			nonTreeEdges[j++] = i;
		}
	}
	/*
	printf("Non tree edges\n");
	for (int i = 0; i < 3 * nonTreeEdgeCount; i+=3) {
		printf("%d %d %d\n", nonTreeEdges[i], nonTreeEdges[i + 1], nonTreeEdges[i + 2]);
	}*/

	int *d_nonTreeEdges, *d_lcaThread, *d_threadEdge, *d_unfinished, *d_auxAdjList, *d_artPoint, *d_rootLCACount;

	cudaMalloc(&d_nonTreeEdges, 3 * nonTreeEdgeCount * sizeof(int));
	cudaMalloc(&d_lcaThread, threadCount * sizeof(int));
	cudaMalloc(&d_threadEdge, edgeCount * sizeof(int));
	cudaMalloc(&d_unfinished, vertexCount * sizeof(int));
	cudaMalloc(&d_auxAdjList, 2 * vertexCount * sizeof(int));
	cudaMalloc(&d_artPoint, vertexCount * sizeof(int));
	cudaMalloc(&d_rootLCACount, sizeof(int));

	cudaMemcpy(d_nonTreeEdges, nonTreeEdges, 3 * nonTreeEdgeCount * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_lcaThread, lcaThread, threadCount * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_threadEdge, threadEdge, edgeCount * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_unfinished, unfinished, vertexCount * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_auxAdjList, auxAdjList, 2 * vertexCount * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_artPoint, artPoint, vertexCount * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rootLCACount, &rootLCACount, sizeof(int), cudaMemcpyHostToDevice);


	lca<<<blocksPerGrid, threadsPerBlock>>>(d_adjList, d_offset, d_parent, d_nonTreeEdges,
											d_unfinished, d_threadEdge, d_lcaThread, d_auxAdjList,
											vertexCount);


	auxGraph<<<blocksPerGrid, threadsPerBlock>>>(d_adjList, d_offset, d_lcaThread, vertexCount, d_rootLCACount, d_auxAdjList);

	int *threadEdge1 = (int *) malloc((edgeCount + vertexCount) * sizeof(int));
	memset(threadEdge1, -1, (edgeCount + vertexCount) * sizeof(int));

	int *d_threadEdge1;
	cudaMalloc(&d_threadEdge1, (edgeCount + vertexCount) * sizeof(int));
	cudaMemcpy(d_threadEdge1, threadEdge1, (edgeCount + vertexCount) * sizeof(int), cudaMemcpyHostToDevice);

	lca1<<<blocksPerGrid, threadsPerBlock>>>(d_adjList, d_offset, d_parent, d_nonTreeEdges,
												d_unfinished, d_threadEdge1, d_lcaThread, d_auxAdjList,
												vertexCount, edgeCount);


	cudaMemcpy(threadEdge1, d_threadEdge1, (edgeCount + vertexCount) * sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(lcaThread, d_lcaThread, threadCount * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(threadEdge, d_threadEdge, edgeCount * sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(unfinished, d_unfinished, vertexCount * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(adjList, d_adjList, edgeCount * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(auxAdjList, d_auxAdjList, 2 * vertexCount * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(artPoint, d_artPoint, vertexCount * sizeof(int), cudaMemcpyDeviceToHost);

/*
	printf("LCA Thread\n");
	for (int i = 0; i < threadCount; i++)
		printf("%d ", lcaThread[i]);

	printf("\nthread Edge\n");
	for (int i = 0; i < edgeCount; i++)
		printf("%d ", threadEdge[i]);

	printf("\n unfinished \n");
	for (int i = 0; i < vertexCount; i++)
		printf("%d ", unfinished[i]);

	printf("\n Adj List\n");
	for (int i = 0; i < edgeCount; i++)
		printf("%d ", adjList[i]);

	printf("\n Aux Adj List \n");
	for (int i = 0; i < 2 * vertexCount; i+=2)
		printf("%d %d\n", auxAdjList[i], auxAdjList[i + 1]);


	printf("\n Art Point \n");
	for (int i = 0; i < vertexCount; i++)
		printf("%d %d\n", i, artPoint[i]);


	printf("\n THREAD EDGE\n");
	for (int i = 0; i < (edgeCount + vertexCount); i++)
		printf("%d ", threadEdge1[i]);

*/
	printf("\n");
	for (int i = 0; i < threadCount; i++) {
		if (threadEdge1[offset[lcaThread[i]]] == -1)
			printf("%d ", lcaThread[i]);
	}
	printf("\n");

	// Free allocated memory on device and host
	cudaFree(d_adjList);
	cudaFree(d_offset);
	cudaFree(d_inpFrontier);
	cudaFree(d_outFrontier);
	cudaFree(d_visited);
	cudaFree(d_treeEdges);
	cudaFree(d_parent);
	cudaFree(d_s1);
	cudaFree(d_s2);
	cudaFree(d_nonTreeEdges);
	cudaFree(d_lcaThread);
	cudaFree(d_threadEdge);
	cudaFree(d_unfinished);
	cudaFree(d_artPoint);
	cudaFree(d_rootLCACount);

	free(inpFrontier);
	free(outFrontier);
	free(visited);
	free(treeEdges);
	free(parent);
	free(nonTreeEdges);
	free(lcaThread);
	free(threadEdge);
	free(unfinished);
	free(artPoint);

    return 0;
}
