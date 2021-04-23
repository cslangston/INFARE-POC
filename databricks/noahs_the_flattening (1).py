# Databricks notebook source
# MAGIC %md
# MAGIC #noahs_the_flattening (1)
# MAGIC Github
# MAGIC 
# MAGIC 
# MAGIC Copy of Noah's Notebook to handle as a baseline for comparison.

# COMMAND ----------

# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, explode
from pyspark.sql.functions import udf
from pyspark.ml.feature import Bucketizer
from pyspark.sql.window import Window
from pyspark.sql import DataFrame

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("spark-avro-json-sample") \
    .config('spark.hadoop.avro.mapred.ignore.inputs.without.extension', 'false') \
    .getOrCreate()

#in_path = 'abfss://gmrch@aabaoriondlsnp.dfs.core.windows.net/raw/ba-n-gmrch-ext-prmry-evhns/ba-n-gmrch-ext-prmry-evh/aod/stage/0/2020/06/02/185651.avro'
in_path = 'abfss://gmrch@aabaoriondlsnp.dfs.core.windows.net/raw/ba-n-gmrch-ext-prmry-evhns/ba-n-gmrch-ext-prmry-evh/aod/test/0/2020/08/16/003107.avro'


#storage->avroA
avroDf = spark.read.format("com.databricks.spark.avro").load(in_path)

#avro->json
jsonRdd = avroDf.select(avroDf.Body.cast("string")).rdd.map(lambda x: x[0])
data = spark.read.json(jsonRdd) # in real world it's better to specify a schema for the JSON
data.printSchema()

# COMMAND ----------

data.show(vertical=True)

# COMMAND ----------

data.count()

# COMMAND ----------

# replace periods with dash
for c in data.columns:
  data = data.withColumnRenamed(c, c.replace(".", "-"))

# COMMAND ----------

data.show(vertical=True)

# COMMAND ----------

flat_cols = []

# COMMAND ----------

fcols = (
  data.withColumn("f_transactionId", col("request-transactionId"))#col("request-transactionId"))
)

# COMMAND ----------

flat_cols.append("f_transactionId") 

# COMMAND ----------

fcols.show(vertical=True)

# COMMAND ----------

fcols.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### request.criteria
# MAGIC >The criteria heading contains most of the information here. There are a few keys which exist only at the top level (accountCodes, displayCurrency, merchandisingPos, and salesCity) so those can just be joined directly with the transactionID as if they were at the same level. Let’s call this our base row. 

# COMMAND ----------

fcols1 = (
fcols.withColumn("f_accountCodes", col("request-criteria-accountCodes")[0]) # would explode but null vals
  .withColumn("f_displayCurrency", col('request-criteria-displayCurrency'))
  .withColumn("f_merchandisingPos", col("request-criteria-merchandisingPos"))
  .withColumn("f_salesCity", col("request-criteria-salesCity"))
)

# COMMAND ----------

fcols1.printSchema()

# COMMAND ----------

flat_cols.extend(["f_accountCodes", "f_displayCurrency", "f_merchandisingPos", "f_salesCity"])

# COMMAND ----------

fcols1.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### request.criteria.passengers
# MAGIC >Then, under “passengers” we find the information of all of the passengers for whom we are requesting prices. We don’t need any of the passenger “profile” details but if we could capture the “id” and value within “ptcs” that would be good. At this point we would duplicate our base row times the number of passengers, which each id-ptcs combo getting its own row. 

# COMMAND ----------

fcols2 = (
  fcols1.withColumn("pass_col", F.explode("request-criteria-passengers"))
  .withColumn("f_ptcs", explode("pass_col.ptcs")) # exploding here because I am assuming never null
  .withColumn("f_passengerId", F.col("pass_col.id"))
)

# COMMAND ----------

fcols2.printSchema()

# COMMAND ----------

fcols2.select("pass_col")

# COMMAND ----------

flat_cols.extend(["f_ptcs", "f_passengerId"])

# COMMAND ----------

fcols2.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### request.criteria.segments
# MAGIC >This is where things get a bit messier. We want to duplicate all of our current rows (base row x pax count) for each segment in the itinerary. Most of the information in criteria:segments could be useful. The “background” information, contained directly in “segments” includes the departure and arrival times, bookingCode, cabin, origin and destination, etc. We need all of these. We’ll also need to label each segment (integers starting at 0) in order to join it with the data in “optionalServiceMatches” (more on that later). 

# COMMAND ----------

fcols3 = (
  fcols2.select(*fcols2.columns, F.posexplode("request-criteria-segments"))
  .withColumnRenamed("pos", "f_seg_idx")
  .withColumnRenamed("col", "f_seg_col")
)
req_seg_cols = ["arrivalTime", "bookingCode", "cabin", "departureTime", "destinationAirport", "fareBasisCode", "flight", "originAirport", "ticketDesignator"]
f_req_seg_cols = ["f_seg_{}".format(c) for c in req_seg_cols]

for c in zip(req_seg_cols, f_req_seg_cols):
  fcols3 = fcols3.withColumn(c[1], fcols3["f_seg_col"][c[0]])
  
fcols3 = fcols3.drop("col")

# COMMAND ----------

flat_cols.extend(f_req_seg_cols)
flat_cols.append("f_seg_idx")

# COMMAND ----------

# MAGIC %md
# MAGIC #### request.criteria.segments.seatMaps
# MAGIC >Also under “segments” is the “seatMaps” key. Under this heading there is some new information and also some repeated information. I spoke with the catalog team and the best they could tell me is that this information is separated and down a level in the off chance that it’s different from the preceding information. The only instance where it would differ that they could think of is in a same-flight number change of gauge flight which, to my knowledge, we don’t operate anymore. However, it’s probably best to keep it separate just in case. Hence, we could label the information directly under “segments” as segment-level information (perhaps “seg_” before the name) and the information under “seatMaps” as the leg information. These will usually be the same. This goes for everything under “seatMaps” (aircraftEquipmentId, cabin, etc) except “seatRows”. For MOST flights we’ll have a number of rows equal to base row x pax count x segment count. In the off chance we do have multiple seatMaps for a segments we’ll have base row x pax count x segment count x leg count. 
# MAGIC 
# MAGIC >The “seatRows” heading (found in criteria:segments:seatMaps) makes up the majority of the rows of the JSON. As near as I can tell, all of the useful information available in this section has already been summarized in the “optionalServiceMatches” section, meaning that everything under “seatRows” can be dropped.

# COMMAND ----------

fcols3a = (
fcols3.select(*fcols3.columns, F.posexplode("f_seg_col.seatMaps"))
  .withColumnRenamed("pos", "f_seg_seatMaps_idx")
  .withColumnRenamed("col", "f_seg_seatMaps_col")
)
req_seg_seatMaps_cols = ["aircraftEquipmentId", "cabin", "departureTime", "destinationAirport", "exitAvailableSeatCount", "flight", "originAirport", "regularAvailableSeatCount"]
f_req_seg_seatMaps_cols = ["f_seg_seatMaps_{}".format(c) for c in req_seg_seatMaps_cols]

# COMMAND ----------

no_seat_maps = fcols3.filter(col("f_seg_col.seatMaps").isNull()) # non seat related ancillary offers (?)

# COMMAND ----------

fcols3.count()

# COMMAND ----------

no_seat_maps.count()

# COMMAND ----------

fcols3a.count()

# COMMAND ----------

for c in zip(req_seg_seatMaps_cols, f_req_seg_seatMaps_cols):
  fcols3a = fcols3a.withColumn(c[1], fcols3a["f_seg_seatMaps_col"][c[0]])

# COMMAND ----------

fcols3a = fcols3a.drop("f_seg_col", "f_seg_seatMaps_col")

# COMMAND ----------

flat_cols.append("f_seg_seatMaps_idx")

# COMMAND ----------

# MAGIC %md
# MAGIC #### request.criteria.optionalServiceMatches
# MAGIC > The final piece of the request record is the “optionalServiceMatches” key, which appears directly under “criteria”. This key contains a long list of catalog KVPs that have been calculated for this request. Here, each key has two parts: “references” can take several values. The first one I see is “isJourney”: true, in which case that key should be broadcast to each segment of the journey. Some “references” may have a pax number which can be joined directly to the paxid mentioned above. Most of the KVPs, however, will have “tripSegmentReference” under “references”. This will have an integer value, starting from 0, recorded as a string, which represents the segment number. So for a one-segment trip each of these keys will appear once and have all 0’s. For a two-segments trip each key will appear twice and have one 0 and one 1. These numbers can be used to join back to the segment information that I mentioned earlier. The other piece of information under each key is the “value” which is what we really care about. It will probably be easiest to make each “key” a column name, populate it with “value” and use “references” to join everything together. I want the vast majority of the keys here, so just keep them all to make things easier.

# COMMAND ----------

fcols4 = (
  fcols3a.withColumn("osm_col", explode("request-criteria-optionalServiceMatches"))
  .withColumn("ref", explode("osm_col.references"))
  .withColumn("isJourney", col("ref.isJourney"))
  .withColumn("paxReference", col("ref.paxReference"))
  .withColumn("tripSegmentReference", col("ref.tripSegmentReference").cast(IntegerType()))
  .withColumn("cat_key", col("osm_col.key"))
  .withColumn("val", when(col("osm_col.`value.numberValue`").isNotNull(), col("osm_col.`value.numberValue`")).otherwise(col("osm_col.`value.stringValue`")))
)

# COMMAND ----------

fcols4.count()

# COMMAND ----------

# attempt to explain osm filtering

# a, [j,k], [1,2], {val:lol, pass:k, trip:2}

# ========================
# a, k, [1,2], {val:lol, pass:k, trip:2}
# a, j, [1,2], {val:lol, pass:k, trip:2}

# ===============================
# a, k, 1, {val:lol, pass:k, trip:2}
# a, k, 2, {val:lol, pass:k, trip:2}
# a, j, 1, {val:lol, pass:k, trip:2}
# a, j, 2, {val:lol, pass:k, trip:2}
# =================================
# a, k, 2, {val:lol, pass:k, trip:2}

# COMMAND ----------

fcols5 = (
  fcols4.filter(col("isJourney").isNotNull() | (col("paxReference") == col("f_passengerId")) | (col("tripSegmentReference") == col("f_seg_idx")))
)

# COMMAND ----------

fcols5.count()

# COMMAND ----------

fcols6 = fcols5.groupby(flat_cols).pivot("cat_key").agg(F.collect_list("val"))

# COMMAND ----------

fcols6.count()

# COMMAND ----------

fcols6 = fcols6.cache()

# COMMAND ----------

fcols3a.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### some column housekeeping

# COMMAND ----------

osm_key_cols = list(set(fcols6.columns) - set(data.columns) - set(flat_cols))

# COMMAND ----------

pre_osm_flat_cols = [ c for c in flat_cols ]

# COMMAND ----------

flat_cols.extend(["f_osm_{}".format(c) for c in osm_key_cols])

# COMMAND ----------

for c1, c2 in zip(osm_key_cols, ["f_osm_{}".format(c) for c in osm_key_cols]):
  fcols6 = fcols6.withColumnRenamed(c1, c2)

# COMMAND ----------

fcols6.printSchema() # request side is now flattened as per Chad's instructions

# COMMAND ----------

# fcols7 = fcols.join(fcols6, on=["f_transactionId"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Response

# COMMAND ----------

resp = fcols3a # start again from the transtaction x passenger x segment x leg level dataframe
row_key = ["f_transactionId", "f_passengerId", "f_seg_idx", "f_seg_seatMaps_idx"]

# COMMAND ----------

resp.count()

# COMMAND ----------

resp = resp.cache()

# COMMAND ----------

resp.select(*row_key).distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### response.segments.seatMaps.seatRows.seats
# MAGIC Finally, we have the top-level “segments” key. Much of this information is the same as was found in the request, but not all of it. Like in the request we have some background information followed by the seat map. While we can delete the “seatMaps” key from the request because the useful information was captured in KVPs that we’re keeping elsewhere that’s not the case for the response. 
# MAGIC 
# MAGIC The “seatRows” heading (found in segments:seatMaps) makes up the majority of the rows of the JSON. Each row is marked by an “id” and the row number as a string (i.e. “id” : “8”), followed by a “seats” key that contains both the features and a “seatId” for each seat letter in that row, just like the request. The difference here is that each specific seat also has a “pricingReferences” key. What I would really like is if we could combine the three letter code from “pricingId” (just the letters, throw away the numbers) with the availability information in “features”. That way, for each request-response pair I would have the complete availability information which I can use to understand the customer’s choice.
# MAGIC 
# MAGIC Essentially, say for an airplane with 4 seat types: MAW, MMM, PAW, and PPP I would have those four rows of the response. Then, within each of those rows I would look at “seatMaps” and summarize the data thusly:
# MAGIC ```
# MAGIC Type	MAW_Count	MAW_Avail	MMM_Count	MMM_Avail	PAW_Count	PAW_Avail	PPP_Count	PPP_Avail
# MAGIC MAW								
# MAGIC MMM								
# MAGIC PAW								
# MAGIC PPP														
# MAGIC ```

# COMMAND ----------

resp.count()

# COMMAND ----------

seatRows = resp.withColumn("maps", explode("response-segments.seatMaps")).filter(col("maps").isNotNull()).withColumn("seatRows", explode("maps.seatRows"))

# COMMAND ----------

seatRows.count()

# COMMAND ----------

seats = seatRows.withColumn("seat_row_id", col("seatRows.id")).withColumn("seat_row_features", col("seatRows.features")).withColumn("seat", explode("seatRows.seats"))

# COMMAND ----------

seats.count()

# COMMAND ----------

seats.select("seat.seatId").take(1)

# COMMAND ----------

seat_info = seats.withColumn("info", explode(F.arrays_zip(col("seat.seatId"), col("seat.features"), col("seat.pricingReferences"))))

# COMMAND ----------

seat_info.select("info").take(1)

# COMMAND ----------

seat_info.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 

# COMMAND ----------

# NOTE: some passenger x segment x leg rows are missing pricingIds
seat_info.withColumn("feats", explode(col("info.1"))).withColumn("is_available", col("feats.description")).filter((col("is_available") == F.lit("SEAT_AVAILABLE")) | (col("is_available") == F.lit("SEAT_OCCUPIED"))).withColumn("seat_pricing_id", col("info.2.pricingId")).filter(col("seat_pricing_id").isNull()).groupby(*row_key).count().count()

# COMMAND ----------

seat_info2 = (
  seat_info
  .withColumn("feats", explode(col("info.1")))
  .withColumn("is_available", col("feats.description"))
  .filter((col("is_available") == F.lit("SEAT_AVAILABLE")) | (col("is_available") == F.lit("SEAT_OCCUPIED")))
  .withColumn("is_available", F.when(col("is_available")==F.lit("SEAT_AVAILABLE"), 1).otherwise(0))
  .withColumn("seat_pricing_id", F.substring(col("info.2.pricingId")[0], 4, 3))
  .filter(col("seat_pricing_id").isNotNull())
)

# COMMAND ----------

maw_paw_table_init = (
  seat_info2
  .groupby(*row_key).pivot("seat_pricing_id").agg(F.collect_list("is_available"))
)

# COMMAND ----------

seat_pricing_ids = [i["seat_pricing_id"] for i in seat_info2.select("seat_pricing_id").distinct().collect()]

# COMMAND ----------

maw_paw_table = maw_paw_table_init
for id in seat_pricing_ids:
  maw_paw_table = (
    maw_paw_table
    .withColumn(f"{id}_count", F.size(maw_paw_table_init[id]))
    .withColumn(f"{id}_available", F.expr(f'AGGREGATE({id},cast(0 as int), (acc, x) -> acc + x)'))
  )
maw_paw_table = maw_paw_table.drop(*seat_pricing_ids)

# COMMAND ----------

maw_paw_table.show(vertical=True)

# COMMAND ----------

maw_paw_table.count()

# COMMAND ----------

fcols6.columns

# COMMAND ----------

pass_seg_leg_rows = fcols6.join(maw_paw_table, on=row_key)

# COMMAND ----------

pass_seg_leg_rows.count()

# COMMAND ----------

maw_paw_table.count()

# COMMAND ----------

f_ = [(col[2:] if col[:2] == "f_" else col) for col in pass_seg_leg_rows.columns ]

# COMMAND ----------

pre = pass_seg_leg_rows

# COMMAND ----------

orig_cols = pre.columns
for new, old in zip(f_, orig_cols):
  pass_seg_leg_rows = pass_seg_leg_rows.withColumnRenamed(old, new)

# COMMAND ----------

psl_cols = pass_seg_leg_rows.columns

# COMMAND ----------

pass_seg_leg_rows.printSchema()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### response.pricings.pricingId
# MAGIC >Under “pricings” there are a number of things I would like to capture, but I’m not sure the best way to do so. Under “pricings” there will be a list of several groupings. Each one will contain “optionalServices”, “passengers”, and “pricingId” keys. The number of groupings that shows up will vary from response to response. Each grouping here represents the seat prices for a specific paid seat type. We have 15 possible seat types, but each aircraft contains only a subset of those 15. The seat type itself is identified by the “pricingId” key. For the file I’m looking at, each “pricingId” value has three numbers and three letters. I don’t know what the numbers mean and I think they can be thrown away (I’ll double check). The letters though represent one of those seat types. One option here would be for me to send you a list of the 15 seat types. You can then create columns for each attribute of each seat type. Those seat types that appear in a “pricingId” will have values and all others can be assigned null values. However, I think the better thing to do is to split the data into multiple rows, one for each seat type. This will be the easiest for using the data found in the other groupings.

# COMMAND ----------

data.printSchema()

# COMMAND ----------

row_key

# COMMAND ----------

os0 = pass_seg_leg_rows.join(
  fcols3a.select(*row_key, "response-pricings"),
  ((pass_seg_leg_rows["transactionId"]==fcols3a['f_transactionId']) &
   (pass_seg_leg_rows['passengerId']==fcols3a['f_passengerId']) & 
   (pass_seg_leg_rows['seg_idx']==fcols3a['f_seg_idx']) & 
   (pass_seg_leg_rows['seg_seatMaps_idx']==fcols3a['f_seg_seatMaps_idx'])),
  "left"
).dropDuplicates(row_key).drop(*row_key) # the test data, for row_key columns, had 2 dupes out of ~2800 rows

# COMMAND ----------

os0.printSchema()

# COMMAND ----------

pass_seg_leg_rows.count()

# COMMAND ----------

os0.count()

# COMMAND ----------

os1 = (
  os0
  .withColumn("os_zip", F.arrays_zip("response-pricings.optionalServices", "response-pricings.pricingId"))
)
os2 = (
  os1
  .select(*os1.columns, F.posexplode("os_zip"))
  .withColumnRenamed("pos", "os_id")
  .withColumnRenamed("col", "os")
)

# COMMAND ----------

os1.printSchema()

# COMMAND ----------

os2.count()

# COMMAND ----------

os2.select("os.1").take(1)

# COMMAND ----------

os_cols = []

# COMMAND ----------

os3 = (
  os2.withColumn("resp_pricing_pricingId", F.substring(col("os.1"), 4, 3))
)


# COMMAND ----------

os_cols.append("resp_pricing_pricingId")

# COMMAND ----------

# MAGIC %md
# MAGIC #### response.pricings.optionalServices
# MAGIC >I do not think that we need the “passengers” columns. Each passenger is seeing all of the responses, so I think we can throw that one away. Under “optionalServices” there is some data worth keeping though. Most of the data under “optionalServices” is simple keys and values, so let’s keep those (the “subcode” is the same seat type code found in pricingId; obviously we only need to keep one). There are three that have levels underneath them: “annotations”, “segments”, and “slices”. By and large the information in “segments” and “slices” is the same as that found in the request record. There are four things I’d like to pull out of there: distance, duration, aircraft, and aircraftFamily. These are different level of specificity of aircraft codes than the one found in request, so we’ll need to come up new names for them. For the majority of cases where we have one leg per segment these should line up easily. On the off case, discussed earlier, of same flight number change of gauge we’ll need to duplicate rows for each leg.

# COMMAND ----------

os4 = os3
resp_optionalServices_cols = ["attributeGroup", "carrier", "commercialName", "concurrence", "displayPrice", "displayTotal", "industryOrCarrier", "mustCheckAvailability", "price", "reasonForIssuance", "refundReissuable", "salePrice", "subcode", "subcodeApplication", "taxes"]
output_resp_optionalServices_cols = ["optionalServices_{}".format(c) for c in resp_optionalServices_cols]
for c1, c2 in zip(output_resp_optionalServices_cols, resp_optionalServices_cols):
  os4 = os4.withColumn(c1, os3["os.0"][c2][0]) # can't use explode here


# COMMAND ----------

os4.printSchema()

# COMMAND ----------

os_cols.extend(output_resp_optionalServices_cols)

# COMMAND ----------

os4.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### response.pricings.optionalServices.annotations
# MAGIC The “annotations” key is the hard one. There is some very useful information here for debugging catalog issues and one of the very first projects that we intend to do with this data involves using these columns. Basically, “annotations” contains three “name”’s for each catalog table that has returned a result for this response. As an example, one of the columns in the sample I’m looking at is called “_goog_ptable_id_1GL”. It has two sister keys, “_goog_ptable_row_1GL” and “_goog_ptable_ver_1GL”. Each one of these “name” values returns a different piece of data about the catalog table named 1GL. It turns out that the only one that is useful for us is the “row” one: “_goog_ptable_row_1GL”. The others (with “id” and “ver”) are for internal Google use and can all be discarded. So for each seat type (or row of our flattened tables, in this case) I have a list of 18 tables (in this example, that’s not necessarily always the number) that I need to record that data from. The other problem is that I need to collect all of the table names. Off the top of my head I’m thinking that you could create a columns called “table_names” and stick a vector of table names inside of it [“1GL”, ….] where you rip off the last three characters of each of the “name” values. Then you could create another column with a vector of the row values (which should all be integers) in the same order as the column vectors. This would account for the fact that different table names will appear for each product. These are also columns that we don’t usually use. We use them enough that we want them but infrequently enough that stuffing them in a vector in a column instead of in their own unique columns makes sense. Of course, I’m open to any other ideas you may have.

# COMMAND ----------

# annotations
annotations = (
  os4.withColumn("resp_pricings_osm_annotations", explode("os.0.annotations"))
  .withColumn("resp_pricings_osm_ann_ele", explode("resp_pricings_osm_annotations"))
  .withColumn("resp_pricings_osm_ann_ele_name", col("resp_pricings_osm_ann_ele.name"))
  .withColumn("resp_pricings_osm_ann_ele_val", col("resp_pricings_osm_ann_ele.values"))
  .filter((F.substring(col("resp_pricings_osm_ann_ele_name"), 1, 17) == F.lit("__goog_ptable_row")) |
          (col("resp_pricings_osm_ann_ele_name") == F.lit("BE 7Day")) | # these last two were specifically requested
          (col("resp_pricings_osm_ann_ele_name") == F.lit("rfisc")) )
  .withColumn("resp_pricings_osm_ann_ele_type", F.when((F.substring(col("resp_pricings_osm_ann_ele_name"), 15, 3) == F.lit("row")), F.substring(col("resp_pricings_osm_ann_ele_name"), 19, 3)).otherwise(col("resp_pricings_osm_ann_ele_name"))) # __goog_ptable_row_??? this will be "row"
  .groupby(*[key[2:] for key in row_key], "os_id") # represents transaction x passenger x segment x leg rows x optionalServices
  .agg(
    F.collect_list(col("resp_pricings_osm_ann_ele_type")).alias("google_tables"),
    F.flatten(F.collect_list(col("resp_pricings_osm_ann_ele_val"))).alias("google_table_values")
  )
)

# fcols_annotations.select("f_google_tables", "f_google_table_values").show(20, False, vertical=True)

# COMMAND ----------

annotations.count()

# COMMAND ----------

flattened = os4.select(*psl_cols, *os_cols, "os_id").join(
  annotations,
  [*[key[2:] for key in row_key], "os_id"]
).drop("os_id")

# COMMAND ----------

flattened.select("optionalServices_taxes").show(vertical=True)

# COMMAND ----------

flattened.printSchema()

# COMMAND ----------

# optional services taxes is a struct, selecting display price attribute to flatten, perhaps there is a more appropriate choice
# |-- optionalServices_taxes: array (nullable = true)
# |    |-- element: struct (containsNull = true)
# |    |    |-- code: string (nullable = true)
# |    |    |-- country: string (nullable = true)
# |    |    |-- displayPrice: string (nullable = true)
# |    |    |-- price: string (nullable = true)
# |    |    |-- salePrice: string (nullable = true)
flattened = flattened.withColumn("optionalServices_taxes", col("optionalServices_taxes.displayPrice"))

# COMMAND ----------

flattened.count()

# COMMAND ----------

flattened.printSchema()

# COMMAND ----------

flattened.show(2, vertical=True)

# COMMAND ----------

#flattened.write.parquet('abfss://gmrch@aabaoriondlsnp.dfs.core.windows.net/raw/ba-n-gmrch-ext-prmry-evhns/ba-n-gmrch-ext-prmry-evh/aod/stage/flat')