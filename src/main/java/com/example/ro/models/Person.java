package com.example.ro.models;


import java.util.Objects;


class Person {
	
	private String name;
	private Gender gender;

	private Person(Builder builder) {
		name = builder.name;
		gender = builder.gender;
	}

	String getName() {
		return name;
	}

	Gender getGender() {
		return gender;
	}

	static final class Builder {

		private String name;
		private Gender gender;

		Builder(String name, Gender gender) {
			Objects.requireNonNull(name);
			this.name = name.toLowerCase();
			this.gender = gender;
		}

		Person build() {
			return new Person(this);
		}
	}
}
