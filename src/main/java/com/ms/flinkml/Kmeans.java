package com.ms.flinkml;

import org.apache.flink.api.common.functions.FlatJoinFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.operators.DataSource;
import org.apache.flink.api.java.operators.IterativeDataSet;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.configuration.Configuration;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * @author 魏孝文
 */
public class Kmeans {
    public static void main(String[] args) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.createLocalEnvironment(1);
        String path = Objects.requireNonNull(Kmeans.class.getClassLoader().getResource("iris.csv")).getPath();
        DataSource<Iris> irisDataSource = env.readCsvFile(path)
                .pojoType(Iris.class, "f1", "f2", "f3", "f4", "category");

        DataSource<Iris> centers = env.fromCollection(initCenters(3));

        IterativeDataSet<Iris> iterateCenters = centers.iterate(10);

        DataSet<Iris> newCenters = irisDataSource.map(new GetNearestCenter("centers"))
                .withBroadcastSet(iterateCenters, "centers")
                .groupBy(x -> x.f1)
                .reduce(new GetNetCenters1())
                .map(new GetNewCenters2());
        
        /*
        设置Kmeans算法停止条件
        一旦每个节点所属类别不再变化，那我们停止聚类
         */
        DataSet<Tuple3<Iris, Integer, Integer>> oldCluster = irisDataSource.map(new GetNearestCenter("centers"))
                .withBroadcastSet(iterateCenters, "centers");
        DataSet<Tuple3<Iris, Integer, Integer>> newCluster = irisDataSource.map(new GetNearestCenter("newCenters"))
                .withBroadcastSet(newCenters, "newCenters");

        DataSet<Object> terminationCriterion = oldCluster.join(newCluster)
                .where(x -> x.f0)
                .equalTo(y -> y.f0)
                .with((FlatJoinFunction<Tuple3<Iris, Integer, Integer>, Tuple3<Iris, Integer, Integer>, Object>) (irisIntegerIntegerTuple3, irisIntegerIntegerTuple32, collector) -> {
                    /*
                    如果使用老的中心点得到的簇id 和 新的中心点得到的簇id不同
                    那么就加入到terminationCriterion中
                    表明循环还不应该结束
                     */
                    if (!irisIntegerIntegerTuple3.f1.equals(irisIntegerIntegerTuple32.f1)) {
                        collector.collect(irisIntegerIntegerTuple3.f0);
                    }
                });


        DataSet<Iris> resultCenters = iterateCenters.closeWith(newCenters, terminationCriterion);

        resultCenters.print();

        irisDataSource.map(new GetNearestCenter("centers"))
                .withBroadcastSet(resultCenters,"centers")
                .print();
    }

    /**
     * 初始化kmeans的中心点
     * @param k kmeans的参数
     * @return 一组初始化的中心点
     */
    static List<Iris> initCenters(int k){
        Random random = new Random();
        List<Iris> centers = new ArrayList<>();
        for (int i = 0;i < k;i++){
            double f1 = random.nextDouble() * 7;
            double f2 = random.nextDouble() * 3;
            double f3 = random.nextDouble() * 6;
            double f4 = random.nextDouble() * 2;
            centers.add(new Iris(f1,f2,f3,f4,String.format("center-%d",i)));
        }
        return centers;
    }

    /**
     * 为每个节点找到句里其最近的中心点
     * result:
     * f0: iris节点
     * f1: 属于哪个cluster
     * f2: 没啥实际意义，为了下面计算中心点做准备
     */
    static class GetNearestCenter extends RichMapFunction<Iris, Tuple3<Iris,Integer,Integer>>{
        String broadcastName;
        List<Iris> centers;

        public GetNearestCenter(String broadcastName) {
            this.broadcastName = broadcastName;
        }

        @Override
        public void open(Configuration parameters) throws Exception {
            super.open(parameters);
            centers = new ArrayList<>();
            centers = getRuntimeContext().getBroadcastVariable(broadcastName);
            System.out.println(broadcastName);
            centers.forEach(System.out::println);
        }

        @Override
        public Tuple3<Iris, Integer, Integer> map(Iris iris) {
            double minDistance = Double.MAX_VALUE;
            int cluster = -1;
            for (Iris center : centers){
                double distance = Math.sqrt(Math.pow(iris.f1 - center.f1 , 2) + Math.pow(iris.f2 - center.f2 , 2) +
                        Math.pow(iris.f3 - center.f3 , 2) + Math.pow(iris.f4 - center.f4 , 2));
                if (distance < minDistance){
                    minDistance = distance;
                    cluster = centers.indexOf(center);
                }
            }
            return Tuple3.of(iris, cluster, 1);
        }
    }


    /**
     * 得到新的中心点
     */
    static class GetNetCenters1 implements ReduceFunction<Tuple3<Iris, Integer, Integer>> {

        @Override
        public Tuple3<Iris, Integer, Integer> reduce(Tuple3<Iris, Integer, Integer> irisTp31, Tuple3<Iris, Integer, Integer> irisTp32) {
            Iris iris1 = irisTp31.f0;
            Iris iris2 = irisTp32.f0;
            Iris result = new Iris(iris1.f1+ iris2.f1, iris1.f2+ iris2.f2, iris1.f3+ iris2.f3, iris1.f4+ iris2.f4, "");
            return Tuple3.of(result, irisTp31.f1, irisTp31.f2+irisTp32.f2);
        }
    }

    static class GetNewCenters2 implements MapFunction<Tuple3<Iris, Integer, Integer>, Iris>{

        @Override
        public Iris map(Tuple3<Iris, Integer, Integer> irisIntegerIntegerTuple3) {
            Iris sum = irisIntegerIntegerTuple3.f0;
            int count = irisIntegerIntegerTuple3.f2;
            return new Iris(sum.f1/count, sum.f2/count, sum.f3/count, sum.f4/count,"");
        }
    }
}
