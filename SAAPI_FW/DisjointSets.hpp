#pragma once
#include <stdlib.h>

class DisjointSet
{
public:
	int N;
	std::vector<int> parent, size;

	DisjointSet(int N)
	{
		this->N = N;
		parent = std::vector<int>(N + 1, 0);
		size = std::vector<int>(N + 1, 1);
	}

	void Union(int x, int y)
	{
		if (x <= 0 || x > N || y <= 0 || y > N)
			return;

		int rx = Find(x), ry = Find(y);

		if (rx != ry)
		{
			if (size[rx] > size[ry])
			{
				parent[ry] = rx;
				size[rx] += size[ry];
			}
			else
			{
				parent[rx] = ry;
				size[ry] += size[rx];
			}
		}
	}

	int Find(int x)
	{
		int initialX = x;
		
		if (x <= 0 || x > N)
			return 0;
		
		while (parent[x] > 0)
			x = parent[x];

		if (parent[initialX])
			parent[initialX] = x;
		
		return x;
	}
};

