use apriori::{apriori, write_rules_to_file, get_good_filename};

fn main() {

    // set min_sup and min_conf
    let min_sup = 0.05;
    let min_conf = 0.3;

    // get good filename based on min_sup and min_conf
    let filename = get_good_filename(min_sup, min_conf);

    // call the apriori function
    let (_fre_sets, association_rules_set) = apriori(min_sup, min_conf, "groceries.csv");

    // write all association rules to file
    write_rules_to_file(&filename, &association_rules_set);
}
