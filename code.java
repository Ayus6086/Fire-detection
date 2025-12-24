import java.util.*;

class Solution {
    public int quickestPaths(int n, int[][] lanes) {
        final long MOD = 1_000_000_007;

        List<int[]>[] g = new ArrayList[n];
        for (int i = 0; i < n; i++) g[i] = new ArrayList<>();
        for (int[] e : lanes) {
            g[e[0]].add(new int[]{e[1], e[2]});
            g[e[1]].add(new int[]{e[0], e[2]});
        }

        long[] dist = new long[n];
        long[] ways = new long[n];
        Arrays.fill(dist, Long.MAX_VALUE);

        PriorityQueue<long[]> pq = new PriorityQueue<>(Comparator.comparingLong(a -> a[1]));
        dist[0] = 0;
        ways[0] = 1;
        pq.add(new long[]{0, 0});

        while (!pq.isEmpty()) {
            long[] cur = pq.poll();
            int u = (int) cur[0];
            long d = cur[1];
            if (d > dist[u]) continue;

            for (int[] e : g[u]) {
                int v = e[0], w = e[1];
                long nd = d + w;

                if (nd < dist[v]) {
                    dist[v] = nd;
                    ways[v] = ways[u];
                    pq.add(new long[]{v, nd});
                } 
                else if (nd == dist[v]) {
                    ways[v] = (ways[v] + ways[u]) % MOD;
                }
            }
        }

        return (int) (ways[n - 1] % MOD);
    }
}